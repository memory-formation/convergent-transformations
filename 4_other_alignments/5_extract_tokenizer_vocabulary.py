"""
5_extract_tokenizer_vocabulary.py

This script extracts token-level representations of image captions from the COCO dataset (used in NSD)
using both:
- Classic NLP tokenization methods (word tokenization, stemming, stopword removal).
- Modern LLM tokenizers (e.g. LLaMA, Bloom, Gemma), via Hugging Face tokenizers.

### Purpose
To compute token-level similarity structures (token RDMs) for images, enabling analysis of how brain regions
align with different lexical segmentations of visual stimuli. These token RDMs are later compared to brain responses
using Representational Similarity Analysis (RSA) or CKA.

**In the final paper**, the tokens generated here are used as inputs to other scripts that compute alignment
via RSA/CKA between fMRI activity and token-level descriptions.

---

### Main Steps
1. **generate_classic_tokenization()**
   - Tokenizes captions using `nltk`: words, stems, with and without stopwords.
   - Produces 4 types: `words`, `stems`, `words_clean`, `stems_clean`.

2. **generate_model_tokens(models)**
   - Tokenizes captions using LLM tokenizers (e.g. LLaMA, Gemma, Bloom).
   - Collapses repeated tokenizers (e.g., different LLaMA variants) to a single representation.
   - Stores output in: `coco-tokens-models.parquet`

3. **compute_token_similarity()**
   - Computes token-level similarity matrices using **Jaccard distance** over token sets.
   - For each subject, session, and ROI, compares the neural responses to the corresponding token 
   similarity matrix using `rsa` and `unbiased_cka`.
   - Stores full results in: `tokenizers-similarities.parquet`


---

### Notes
- The Jaccard similarity matrix is used as a semantic proxy for token overlap.
- Tokenizers are reduced to canonical representations to avoid duplication across similar models.
- Downstream alignment analyses (e.g. comparison to brain data using RSA) are run in a separate script.

"""

import pandas as pd
from convergence.nsd import get_resource, get_index
import os
from pathlib import Path
from convergence.feature_extraction.llm import load_tokenizer
from dmf.io import load
import numpy as np
from tqdm import tqdm

import torch
from dmf.alerts import send_alert, alert

from tqdm import trange, tqdm
from convergence.nsd import get_subject_roi
from convergence.metrics import measure

from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import gc


def generate_classic_tokenization():
    stemmer = PorterStemmer()

    df_captions = get_resource("coco-captions")
    df_captions["caption"] = df_captions["captions"].str.split(";").str[0].str.strip()
    df_captions = df_captions[["nsd_id", "caption"]]

    word_tokenize = TreebankWordTokenizer().tokenize
    word_tokens = (
        df_captions["caption"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(".", " ")
        .str.replace(",", " ")
        .str.strip()
        .apply(word_tokenize)
    )

    df_captions["words"] = word_tokens

    df_captions["stems"] = df_captions["words"].apply(
        lambda tokens: [stemmer.stem(token) for token in tokens]
    )

    # Remove stopwords

    stop_words = set(stopwords.words("english"))
    # Remove 's also
    stop_words.add("'s")

    df_captions["words_clean"] = df_captions["words"].apply(
        lambda tokens: [token for token in tokens if token not in stop_words]
    )

    df_captions["stems_clean"] = df_captions["stems"].apply(
        lambda tokens: [token for token in tokens if token not in stop_words]
    )

    df_captions = df_captions.drop(columns=["caption"])

    # unpivot. Words, stems, words_clean, stems_clean
    df_captions = df_captions.melt(
        id_vars=["nsd_id"], var_name="type", value_name="tokens"
    )
    df_captions = df_captions.reset_index(drop=True)
    df_captions["type"] = df_captions["type"].astype("str").astype("category")

    df_captions.to_parquet("coco-classic-tokenizer.parquet", index=False)


def tokenize(model_name: str, captions: list[str]) -> np.ndarray:
    tokenizer = load_tokenizer(model_name)
    tokens = tokenizer(
        captions,
        return_tensors="np",
        return_special_tokens_mask=True,
        padding="longest",
        padding_side="left",
        return_attention_mask=False,
    )
    input_ids = tokens["input_ids"]
    special_tokens_mask = tokens["special_tokens_mask"]
    input_ids[special_tokens_mask == 1] = -1
    return input_ids


def token_to_lists(tokens: np.ndarray) -> list[list[int]]:
    token_list = list(
        map(
            lambda tokenized_caption: tokenized_caption[
                tokenized_caption != -1
            ].tolist(),
            tokens,
        )
    )
    return token_list


def tokenized_captions(model_name: str, df_captions: pd.DataFrame) -> pd.DataFrame:
    captions = df_captions["caption"].tolist()
    tokens = tokenize(model_name, captions)
    token_list = token_to_lists(tokens)

    df_tokens = df_captions[["nsd_id"]].copy()
    df_tokens["tokens"] = token_list
    df_tokens["model"] = model_name

    return df_tokens


def jaccard_distance(sentence_list: list[list[str]]) -> np.ndarray:
    n = len(sentence_list)
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            set1 = set(sentence_list[i])
            set2 = set(sentence_list[j])
            jaccard[i, j] = len(set1.intersection(set2)) / len(set1.union(set2))
            jaccard[j, i] = jaccard[i, j]
    return jaccard


def generate_model_tokens(models):

    df_captions = get_resource("coco-captions")
    df_captions["caption"] = df_captions["captions"].str.split(";").str[0].str.strip()

    df_tokens = []
    for model in tqdm(models, leave=False):
        df_tokens.append(tokenized_captions(model, df_captions))
    df_tokens = pd.concat(df_tokens)

    repeated_tokenizers = {
        "bigscience/bloomz-560m": "bigscience/bloomz-560m",
        "bigscience/bloomz-1b1": "bigscience/bloomz-560m",
        "bigscience/bloomz-1b7": "bigscience/bloomz-560m",
        "bigscience/bloomz-3b": "bigscience/bloomz-560m",
        "bigscience/bloomz-7b1": "bigscience/bloomz-560m",
        "openlm-research/open_llama_3b": "openlm-research/open_llama_3b",
        "openlm-research/open_llama_7b": "openlm-research/open_llama_3b",
        "openlm-research/open_llama_13b": "openlm-research/open_llama_3b",
        "huggyllama/llama-7b": "meta-llama/Llama-2-7b",
        "huggyllama/llama-13b": "meta-llama/Llama-2-7b",
        "meta-llama/Llama-2-7b": "meta-llama/Llama-2-7b",
        "meta-llama/Llama-2-13b": "meta-llama/Llama-2-7b",
        "meta-llama/Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-3.1-8B": "meta-llama/Meta-Llama-3-8B",
        "google/gemma-2-2b": "google/gemma-2-2b",
        "google/gemma-2-9b": "google/gemma-2-2b",
    }

    for model, repeated_tokenizer in repeated_tokenizers.items():
        tokens_model = df_tokens.query(f"model=='{model}'").tokens.astype("str")
        tokens_repeated = df_tokens.query(
            f"model=='{repeated_tokenizer}'"
        ).tokens.astype("str")
        assert (
            tokens_model == tokens_repeated
        ).all(), f"Tokens are not the same for {model} and {repeated_tokenizer}"

    df_tokens = df_tokens[
        df_tokens.model.isin(list(repeated_tokenizers.values()))
    ].copy()

    df_tokens.model = df_tokens.model.astype("str").astype("category")
    df_tokens = df_tokens.reset_index(drop=True)
    df_tokens = df_tokens.copy()

    reverse_tokens = {}
    for k, v in repeated_tokenizers.items():
        reverse_tokens.setdefault(v, [])
        reverse_tokens[v].append(k)

    for k, v in reverse_tokens.items():
        reverse_tokens[k] = ";".join(v)

    df_tokens["models"] = df_tokens["model"].map(reverse_tokens)

    df_tokens.models = df_tokens.models.astype("str").astype("category")

    df_tokens.to_parquet("coco-tokens-models.parquet", index=False)




def get_subject_tokens(df_subject: int, df_tokens: pd.DataFrame, tokenizers) -> pd.DataFrame:
    matrices_dict = {}
    sessions = df_subject["session"].unique()
    pbar = tqdm(total=len(sessions) * len(tokenizers), leave=False, position=1, desc="Tokenizing")
    for session in sessions:
        matrices_dict[session] = {}
        nsd_id = df_subject.query(f"session=={session}")["nsd_id"].values.tolist()
        for tokenizer in tokenizers:
            tokens = df_tokens.query(f"model=='{tokenizer}'").tokens.values
            tokens = tokens[nsd_id]
            jaccard_matrix = jaccard_distance(tokens)
            jaccard_matrix = torch.tensor(jaccard_matrix, dtype=torch.float32, device="cuda")
            matrices_dict[session][tokenizer] = jaccard_matrix
            pbar.update(1)
    pbar.close()
    return matrices_dict

@alert
def compute_token_similarity():
    df_tokens_classic = pd.read_parquet("coco-classic-tokenizer.parquet")
    df_tokens_models = pd.read_parquet("coco-tokens-models.parquet")
    df_tokens_classic.rename(columns={"type": "model"}, inplace=True)

    df_tokens_models = df_tokens_models[["nsd_id", "model", "tokens"]]
    df_tokens_classic = df_tokens_classic[["nsd_id", "model", "tokens"]]
    df_tokens = pd.concat([df_tokens_classic, df_tokens_models])
    df_tokens.model = df_tokens.model.astype("str").astype("category")
    tokenizers = list(df_tokens.model.unique())
    

    df_stimulus = get_index("stimulus")
    df_similarities = []
    for subject in trange(1, 9, leave=False, position=0):
        send_alert(f"Subject {subject}")
        df_subject = df_stimulus.query(f"subject=={subject} and exists")
        subject_tokens = get_subject_tokens(df_subject, df_tokens, tokenizers)
        sessions = df_subject["session"].unique()
        
        for roi in trange(1, 361, leave=False, position=1):
            roi_betas = get_subject_roi(subject=subject, roi=roi)
            for session in tqdm(sessions, leave=False, position=2):
                df_session_subject = df_subject.query(f"session=={session}")
                subject_index = df_session_subject["subject_index"].values.tolist()

                session_betas_roi = roi_betas[subject_index]
                a, b = np.quantile(session_betas_roi, [0.01, 0.99])
                session_betas_roi = np.clip(session_betas_roi, a, b)
                session_betas_roi = torch.tensor(session_betas_roi, dtype=torch.float32, device="cuda")
                
                for tokenizer in tqdm(tokenizers, leave=False, position=3):
                    jaccard_matrix = subject_tokens[session][tokenizer]
                    for metric in ["rsa", "unbiased_cka"]:

                        r = measure(metric, session_betas_roi, jaccard_matrix)
                        df_similarities.append(
                            {
                                "subject": int(subject),
                                "session": int(session),
                                "roi": int(roi),
                                "tokenizer": tokenizer,
                                "metric": metric,
                                "score": r,
                            }
                        )
                del session_betas_roi
            del roi_betas
    del subject_tokens
    gc.collect()
    torch.cuda.empty_cache()

    df_similarities = pd.DataFrame(df_similarities)
    df_similarities.subject = df_similarities.subject.astype("int16").astype("category")
    df_similarities.session = df_similarities.session.astype("int16").astype("category")
    df_similarities.roi = df_similarities.roi.astype("int16").astype("category")
    df_similarities.tokenizer = df_similarities.tokenizer.astype("str").astype(
        "category"
    )
    df_similarities.score = df_similarities.score.astype("float32")
    df_similarities.metric = df_similarities.metric.astype("str").astype("category")
    df_similarities.to_parquet("tokenizers-similarities.parquet", index=False)

if __name__ == "__main__":

    models = load("models.yml")
    models = models["language"]

    generate_model_tokens(models)
    generate_classic_tokenization()
    compute_token_similarity()

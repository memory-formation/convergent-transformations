"""
Compute alignment between NSD brain responses and tokenized image captions.

This script:
- Compares one-hot token embeddings (from multiple tokenizers) with brain ROI responses.
- Operates per session, subject, ROI (1–360 or joined hemispheres), and tokenizer.
- Uses a similarity metric (default: unbiased CKA) to quantify alignment.

Inputs:
- Brain betas per subject and ROI (via `get_subject_roi`).
- Tokens from different tokenizers (LLaMA-2, LLaMA-3, OpenLLaMA, classic).
- NSD metadata (stimulus index, subject/session mappings).

Outputs:
- A `.parquet` file with alignment scores per ROI, session, subject, and tokenizer.

Note:
- Token embeddings are one-hot encoded per image (optionally with counts).
- Brain signals are quantile-clipped and normalized before comparison.
"""

import pandas as pd
from tqdm import trange, tqdm
import numpy as np
from convergence.nsd import get_index, get_subject_roi
from convergence.metrics import measure
import torch
from dmf.alerts import alert, send_alert


def one_hot_encode_list_of_lists(list_of_lists, counts=False):
    """
    Given a list of lists of token IDs, return a NumPy array
    with one-hot encoding.

    :param list_of_lists: List of lists, where each sub-list
                          is a sequence of token IDs.
    :return: A NumPy array of shape (num_samples, num_unique_tokens)
             containing the one-hot encodings.
    """
    # 1) Find all unique tokens in the entire dataset
    unique_tokens = set()
    for tokens in list_of_lists:
        unique_tokens.update(list(tokens))

    # 2) Convert to a sorted list (or keep in any fixed order you like)
    unique_tokens = sorted(unique_tokens)

    # 3) Create a mapping from token -> index
    token_to_index = {token: i for i, token in enumerate(unique_tokens)}

    # 4) Initialize the one-hot matrix
    num_samples = len(list_of_lists)
    num_unique_tokens = len(unique_tokens)
    one_hot_matrix = np.zeros((num_samples, num_unique_tokens), dtype=int)

    # 5) Populate the one-hot matrix
    for i, tokens in enumerate(list_of_lists):
        for token in tokens:
            idx = token_to_index[token]
            if counts:
                one_hot_matrix[i, idx] += 1
            else:
                one_hot_matrix[i, idx] = 1

    return one_hot_matrix


def load_tokens():
    # Could be cleaner...
    df_tokens = pd.read_parquet(
        "/mnt/tecla/Results/convergence/tokens/coco-tokens-models.parquet"
    )
    df_classic = pd.read_parquet(
        "/mnt/tecla/Results/convergence/tokens/coco-classic-tokenizer.parquet"
    )
    df_tokens = df_tokens[
        df_tokens.model.isin(
            [
                "openlm-research/open_llama_3b",
                "meta-llama/Meta-Llama-3-8B",
                "meta-llama/Llama-2-7b",
            ]
        )
    ]
    df_tokens = df_tokens.rename(columns={"model": "tokenizer"}).drop(
        columns=["models"]
    )
    df_classic = df_classic.rename(columns={"type": "tokenizer"})
    df = pd.concat([df_tokens, df_classic])
    return df


def prepare_betas(
    betas,
    subject_index: list[int],
    q: float = 0.003,
) -> torch.tensor:

    betas = betas[subject_index].astype("float32")
    q0, q1 = np.quantile(betas, [q, 1 - q])
    betas = np.clip(betas, q0, q1)
    betas = (betas - q0) / (q1 - q0)

    betas = torch.tensor(betas, device="cuda", dtype=torch.float32)
    return betas


def prepare_tokens(df, tokenizer, nsd_ids, counts=False):
    df_tokenizer = df.query(f"tokenizer == '{tokenizer}'")
    assert len(df_tokenizer) == 73000
    df_tokenizer = df_tokenizer.sort_values(by="nsd_id").reset_index(drop=True)
    tokens = df_tokenizer.iloc[nsd_ids].tokens.tolist()
    matrix = one_hot_encode_list_of_lists(tokens, counts=counts)
    matrix = torch.tensor(matrix, device="cuda", dtype=torch.float32)
    return matrix


@alert
def compute_alignment(df, output_file, join_hemispheres=True, metric="unbiased_cka"):
    df_nsd = get_index("stimulus")

    results = []
    tokenizers = list(df.tokenizer.unique())
    for subject in trange(1, 9, position=0, leave=False):
        send_alert(f"Processing subject {subject} - computing token alignment")
        sessions = df_nsd.query(f"subject == {subject} and exists").session.unique()
        total_rois = 180 if join_hemispheres else 360
        for roi in trange(1, total_rois + 1, position=1, leave=False):
            roi_n = roi
            if join_hemispheres:
                roi = [roi, roi + 180]
            betas = get_subject_roi(subject, roi)

            for session in tqdm(sessions, position=2, leave=False):
                df_session = df_nsd.query(
                    f"subject == {subject} and session == {session} and exists"
                )
                nsd_ids = df_session.nsd_id.tolist()
                subject_indexes = df_session.subject_index.tolist()
                betas_session = prepare_betas(betas, subject_indexes)

                for tokenizer in tokenizers:
                    token_matrix = prepare_tokens(df, tokenizer, nsd_ids)
                    r = measure(metric, betas_session, token_matrix)
                    results.append(
                        {
                            "subject": subject,
                            "session": session,
                            "roi": roi_n,
                            "tokenizer": tokenizer,
                            "score": r,
                            "metric": metric,
                        }
                    )

    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_file)


if __name__ == "__main__":
    df = load_tokens()
    output_file = "token_cka_similarity_separated.parquet"

    metric = "unbiased_cka" # rsa, cka, unbiased_cka
    compute_alignment(df, output_file, join_hemispheres=False, metric=metric)

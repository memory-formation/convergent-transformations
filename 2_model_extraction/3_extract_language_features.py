"""
Script for extracting activations from language models using image captions.

This script loads a set of captions (generated or annotated), and processes them through 
a list of transformer-based language models (defined in `models.yml`) to extract 
layer-wise activations. These features are saved as `.pt` tensors for downstream analysis 
(e.g., representational alignment with brain data).

Usage:
    python extract_language_features.py --captions <path/to/captions.csv> \
                                        --prompt_name <prompt_id> \
                                        --dataset_name <dataset_id> \
                                        [--force_remake] \
                                        [--batch_size <int>] \
                                        [--pool <avg|cls>] \
                                        [--output_dir <path>] \
                                        [--unpretrained]

Arguments:
    --captions: Path to a CSV or Parquet file containing the captions. Must include a 'caption' or 'captions' column.
    --prompt_name: Identifier for the prompt used to generate or describe the captions (e.g. 'coco', 'pixtral').
    --dataset_name: Name of the dataset (used for output naming and bookkeeping).
    --force_remake: If set, recomputes features even if output exists.
    --batch_size: Batch size used during model inference (default: 12).
    --pool: Pooling strategy for model outputs ('avg' or 'cls', default: 'avg').
    --output_dir: Directory to save output `.pt` files (default: './results/features').
    --unpretrained: If set, uses untrained (randomly initialized) model weights.

Output:
    For each model, saves a `.pt` file with a tensor of shape `[n_texts x n_layers x n_dim]`
    and accompanying metadata into the specified output directory.

Requirements:
    - Captions must be pre-generated and cleaned if necessary.
    - `models.yml` must include a 'language' section listing the model names or paths.

"""

import logging
import argparse
import pandas as pd
from pathlib import Path
from dmf.io import load

from convergence.feature_extraction.llm import extract_llm_features

MODELS_FILE = "models.yml"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=str, required=True)
    parser.add_argument("--prompt_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--force_remake", action="store_true")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--pool", type=str, default="avg")
    parser.add_argument("--output_dir", type=str, default="./results/features")
    parser.add_argument("--unpretrained", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("extraction")

    models = load(MODELS_FILE)
    llm_models = models["language"]

    logging.info(f"Using ({len(llm_models)}) models: {llm_models}")

    # Load the captions dataset
    captions = Path(args.captions)
    if not captions.exists():
        raise FileNotFoundError(f"Captions file not found: {args.captions}")
    if captions.suffix == ".csv":
        df = pd.read_csv(args.captions)
    elif captions.suffix == ".parquet":
        df = pd.read_parquet(args.captions)
    else:
        raise ValueError(f"Unsupported file format: {captions.suffix}")

    # In coco there are several captions per image, we take the first one
    if "captions" in df.columns:
        logging.info("Using 'captions' column for captions.")
        df["caption"] = df["captions"].str.split(";").str[0].str.strip()

    texts = df["caption"].tolist()
    prompt = args.prompt_name
    dataset_name = args.dataset_name

    logging.info(f"Using prompt name: {prompt}")
    logging.info(f"Using dataset name: {dataset_name}")
    logging.info(f"Number of captions: {len(texts)}")
    logger.info(f"Example first caption: {texts[0]}")

    pool = args.pool
    output_dir = args.output_dir
    batch_size = args.batch_size
    force_remake = args.force_remake

    extract_llm_features(
        filenames=llm_models,
        output_dir=output_dir,
        dataset=texts,
        dataset_name=dataset_name,
        texts=texts,
        subset="all",
        pool=pool,
        force_remake=force_remake,
        force_download=False,
        batch_size=batch_size,
        prompt=prompt,
        caption_idx=0,
        qlora=False,
        from_init=args.unpretrained,
    )

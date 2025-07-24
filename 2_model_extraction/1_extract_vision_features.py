"""
extract_vision_features.py

Script to extract layer-wise features from vision transformer models for visual datasets.

This script loads image stimuli from a specified dataset (e.g., NSD, THINGS, or BOLD5000),
processes them through a list of vision models (defined in `models.yml`), and saves the
activation features from each transformer block.

Usage:
    python extract_vision_features.py --dataset <nsd|things|bold5000> \
                                      [--force_remake] \
                                      [--batch_size <int>] \
                                      [--pool <cls|avg>] \
                                      [--output_dir <path>] \
                                      [--unpretrained]

Arguments:
    --dataset: Dataset name (must be one of: nsd, things, bold5000).
    --force_remake: If set, recomputes features even if output files exist.
    --batch_size: Batch size used during model inference (default: 4).
    --pool: Pooling strategy to summarize token outputs ('cls' or 'avg', default: 'cls').
    --output_dir: Directory to save the output `.pt` feature files (default: './results/features').
    --unpretrained: If set, uses untrained (randomly initialized) model weights instead of pretrained ones.

Requirements:
    - The `models.yml` file must define a list of vision models under the 'vision' key.
    - The appropriate `load_dataset` function must be implemented and return a list of images for the specified dataset.
    - Model features are saved as `.pt` files containing tensors of shape `[n_images × n_layers × n_dim]`.

"""

import argparse
import os
import logging

from convergence.feature_extraction.lvm import extract_lvm_features

from dmf.io import load

if __name__ == "__main__":

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("extraction")

    # Move working directory to the script location parent directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--force_remake", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pool", type=str, default="cls")
    parser.add_argument("--output_dir", type=str, default="./results/features")
    parser.add_argument("--unpretrained", action="store_true")
    parser.add_argument("--dataset", type=str, choices=["nsd", "things", "bold5000"])
    args = parser.parse_args()
    dataset = args.dataset

    # Get list of Vision models from the configuration file
    lvm_models = load("models.yml")["vision"]

    # If load unpretrained models for untrained models extraction
    pretrained = not args.unpretrained

    if dataset == "nsd":
        from convergence.nsd import load_dataset
        dataset_name = "nsd"
    elif dataset == "things":
        from convergence.things import load_dataset
        dataset_name = "things"
    elif dataset == "bold5000":
        from convergence.bold5000 import load_dataset
        dataset_name = "bold5000"

    
    dataset = load_dataset()
    pool = args.pool
    output_dir = args.output_dir
    batch_size = args.batch_size
    force_remake = args.force_remake

    logger.info(f"Extracting features for dataset: {dataset_name}")
    logger.info(f"Using ({len(lvm_models)}) models: {lvm_models}")
    logger.info(f"Number of images in dataset: {len(dataset)}")

    extract_lvm_features(
        lvm_models,
        dataset,
        pool=pool,
        output_dir=output_dir,
        dataset_name=dataset_name,
        force_remake=force_remake,
        batch_size=batch_size,
        pretrained=pretrained,
    )

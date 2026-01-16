"""
rsa_nsd_subject_model_alignment.py

Compute ROI-level Representational Similarity Analysis (RSA) between fMRI data from the 
Natural Scenes Dataset (NSD) subjects and deep neural network features in the shared1000 subset.

This script performs subject-model RSA by aligning RDMs from:
- Subject-specific fMRI activations (within ROI)
- Model feature vectors (across layers)

Key Features:
- Optionally caches subject features for reuse
- Supports both separated (360) and joined (180) hemisphere ROI configurations
- Efficient GPU-based computation
- Optional permutation testing for significance estimation

Arguments:
    --output_filename           Output filename (default: rsa_optimized_model_subject.parquet)
    --join_hemispheres          Merge hemispheres into 180 ROIs (default: False)
    --cache_folder              Folder to cache subject tensors (optional)
    --n_permutations            Number of permutations for null distribution (optional)
    --permutations_folder       Folder to store/retrieve permutation arrays (default: permutations)

Expected Inputs:
- NSD beta responses (via `get_subject_roi` from `convergence`)
- Model feature files: `.pt` files with structure [n_stim x n_layers x n_features]
- Stimulus metadata from `get_resource("stimulus")`

Outputs:
- One `.parquet` file per model, containing:
    Columns: ['subject', 'roi', 'layer', 'similarity', 'model', 'join_hemispheres']

Typical Workflow:
1. Load NSD betas and model features
2. Compute flat RDMs per ROI and layer
3. Calculate cosine RSA between subject and model RDMs

Example:
    python rsa_nsd_subject_model_alignment.py --join_hemispheres

"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import trange, tqdm

import gc
import pandas as pd

from dmf.alerts import alert, send_alert
from convergence.nsd import get_subject_roi, get_resource


DEVICE = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cross-subject similarities with various metrics and configurations."
    )

    # Argument for output filename
    parser.add_argument(
        "--output_filename",
        type=str,
        default="rsa_optimized_model_subject.parquet",
        help="Output filename for the results. Default is based on the shift value.",
    )
    parser.add_argument(
        "--join_hemispheres",
        action="store_true",
    )

    # Add a cache_folder argument. str. Optional
    parser.add_argument(
        "--cache_folder",
        type=str,  # Filter those that end with -pixtral.pt
        default=None,
    )
    

    # Parse the arguments
    args = parser.parse_args()
    if args.output_filename is None:
        joined_suffix = "joined" if args.join_hemispheres else "separated"
        diagonal_suffix = "_diagonal" if args.diagonal else ""
        args.output_filename = (
            f"rsa_subject_model_shared_alignment_{joined_suffix}_{args.shift}{diagonal_suffix}"
        )

    if not args.output_filename.endswith(".parquet"):
        args.output_filename += ".parquet"

    return args


def load_model_paths():
    models_folder = Path("/mnt/tecla/Results/convergence/features/nsd/all")
    models = list(models_folder.glob("*.pt"))
    models = [model for model in models if not model.stem.endswith("-pixtral")]
    return models


def load_model_features(model_path: Path):
    features = torch.load(model_path, weights_only=True)
    features = (
        features["feats"].to(torch.float32).numpy()
    )  # n_stim x n_layers x n_features
    return features


def create_flat_normalize_rdm(
    rdm: torch.Tensor, triu_indices: torch.Tensor = None
) -> torch.Tensor:
    if triu_indices is None:
        triu_indices = torch.triu_indices(rdm.size(0), rdm.size(0), offset=1)
    rdm_flat = rdm[triu_indices[0], triu_indices[1]]
    # rdm_flat = (rdm_flat - rdm_flat.mean()) / rdm_flat.norm() # <- This is for using cosine similarity
    rdm_flat = rdm_flat - rdm_flat.mean()
    rdm_flat /= rdm_flat.norm()
    return rdm_flat


def compute_flat_rdm(betas_subset: np.ndarray, q=0.003):
    a, b = np.quantile(betas_subset, [q, 1 - q])
    betas_subset = np.clip(betas_subset, a, b)

    betas_subset = (betas_subset - a) / (b - a)
    betas_subset = torch.tensor(betas_subset, device=DEVICE, dtype=torch.float32)

    # Compute RDM
    betas_subset = betas_subset - betas_subset.mean(dim=1, keepdim=True)
    betas_subset = torch.nn.functional.normalize(betas_subset, dim=1)
    rdm = 1 - torch.mm(betas_subset, betas_subset.t())
    return create_flat_normalize_rdm(rdm)


def prepare_subject_features(subject, df_stim, join_hemisphere: bool):

    total_rois = 180 if join_hemisphere else 360
    roi_session_rdms = []

    for roi in trange(
        1, total_rois + 1, desc="Preparing ROIs", position=2, leave=False
    ):
        roi_betas = get_subject_roi(
            subject, roi if not join_hemisphere else [roi, roi + 180]
        )
        
        session_subject_index = df_stim.subject_index.values
        flat_rdm = compute_flat_rdm(roi_betas[session_subject_index])
        roi_session_rdms.append(flat_rdm)

    # Stack as a 2D tensor of (n_roi x n_flat_rdm_shape)
    features = torch.stack(roi_session_rdms)

    return features


def prepare_model_features(model_features: np.ndarray, df_stim):

    n_layers = model_features.shape[1]
    layer_session_rdms = []

    for layer in trange(n_layers, desc="Preparing Layers", position=2, leave=False):
        
        session_nsd_ids = df_stim.nsd_id.values
        session_features = model_features[session_nsd_ids, layer, :]
        flat_rdm = compute_flat_rdm(session_features)
        layer_session_rdms.append(flat_rdm)

    # Stack as a 3D tensor of (n_layers x n_flat_rdms)
    features = torch.stack(layer_session_rdms)

    return features


def unravel_tensor(
    tensor: np.ndarray, subject: int, model_name: str, join_hemispheres: bool
) -> pd.DataFrame:
    # n_roi x n_layer.
    data = []
    n_roi, n_layer = tensor.shape
    for roi in range(n_roi):
        for layer in range(n_layer):
                data.append(
                    {
                        "roi": roi + 1,  # ROI is indexed from 1
                        "layer": layer,  # Layer is indexed from 0
                        "similarity": tensor[roi, layer],
                    }
                )

    df = pd.DataFrame(data)
    df["subject"] = subject
    df["model"] = model_name
    df["join_hemispheres"] = join_hemispheres
    df["join_hemispheres"] = df["join_hemispheres"].astype("bool")
    df.subject = df.subject.astype("uint8")  # 1-8
    df.roi = df.roi.astype("uint16")  # 1-360
    df.layer = df.layer.astype("uint8")  # 0-40
    df.similarity = df.similarity.astype("float32")
    df.model = df.model.astype("string").astype("category")
    return df


def cached_prepare_subject_features(
    subject, df_stim, join_hemisphere: bool, cache_folder: str
):

    if cache_folder is not None:
        cache_folder = Path(cache_folder)
        cache_folder.mkdir(exist_ok=True, parents=True)
        cache_file = cache_folder / f"subject_{subject}_features.npy"
        if cache_file.exists():
            subject_features = torch.from_numpy(np.load(cache_file)).to(DEVICE)
        else:
            subject_features = prepare_subject_features(
                subject, df_stim, join_hemisphere
            )
            np.save(cache_file, subject_features.cpu().numpy())
    else:
        subject_features = prepare_subject_features(subject, df_stim, join_hemisphere)

    return subject_features

def compare_subject_model(
    model_features: np.ndarray,
    subject: int,
    join_hemisphere: bool,
    model_name: str,
    cache_folder: str,
):
    df_stim = get_resource("stimulus").query("subject == @subject and exists and shared")
    subject_features = cached_prepare_subject_features(
        subject, df_stim, join_hemisphere, cache_folder
    ) # n_roi x n_flat_rdm
    model_features = prepare_model_features(model_features, df_stim) # n_layers x n_flat_rdm

    # Compute RSA tensor subject_features x model_features^T
    rsa_tensor = torch.mm(
        subject_features, model_features.t()
    ).cpu().numpy()  # n_roi x n_layer
    return unravel_tensor(
        tensor=rsa_tensor,
        subject=subject,
        model_name=model_name,
        join_hemispheres=join_hemisphere,
    )


@alert
def main():

    args = parse_args()
    output_filename = Path(args.output_filename)
    models = load_model_paths()
    join_hemisphere = args.join_hemispheres
    cache_folder = args.cache_folder
    n_subjects = 8

    for model_path in tqdm(models, position=0, desc="Model", leave=False):
        results = []
        model_name = model_path.stem
        filename_model = Path(f"{model_name}_{output_filename.name}")
        if filename_model.exists():
            print("Skipping model", model_name)
            continue

        model_features = load_model_features(model_path)

        send_alert(f"Processing model {model_name}")
        for subject in trange(
            1, n_subjects + 1, position=1, desc="Subject", leave=False
        ):
            df_model_subject = compare_subject_model(
                model_features=model_features,
                subject=subject,
                join_hemisphere=join_hemisphere,
                model_name=model_name,
                cache_folder=cache_folder,
            )
            results.append(df_model_subject)
            gc.collect()
            torch.cuda.empty_cache()

        results = pd.concat(results)
        results.to_parquet(filename_model, index=False)
        del results
        gc.collect()


if __name__ == "__main__":
    main()

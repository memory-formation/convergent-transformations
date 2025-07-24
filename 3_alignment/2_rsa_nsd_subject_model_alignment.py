"""
rsa_nsd_subject_model_alignment.py

Compute ROI-level Representational Similarity Analysis (RSA) between fMRI data from the 
Natural Scenes Dataset (NSD) subjects and deep neural network features, optionally across 
multiple sessions and with permutation-based null distributions.

This script performs subject-model RSA by aligning RDMs from:
- Subject-specific fMRI activations (within ROI and session)
- Model feature vectors (across layers and sessions)

Key Features:
- Computes session-specific ROI x model-layer RSA scores
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
    Columns: ['subject', 'roi', 'layer', 'session', 'similarity', 'model', 'join_hemispheres']
    If permutations are enabled: also saves `.npy` with shape [n_perm, n_roi, n_layer, n_session]

Typical Workflow:
1. Load NSD betas and model features
2. Compute flat RDMs per ROI x session (subject) and per layer x session (model)
3. Calculate cosine RSA between subject and model RDMs
4. Optionally compute null distributions via batched permutations

Example:
    python rsa_nsd_subject_model_alignment.py --join_hemispheres --n_permutations 1000

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
from convergence.permutations import (
    generate_batched_permutations,
)


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

    parser.add_argument(
        "--n_permutations",
        type=int,
        default=None,
        help="Number of permutations to compute the null distribution.",
    )
    parser.add_argument(
        "--permutations_folder",
        type=str,
        default="permutations",
        help="Folder to store the permutations",
    )

    # Parse the arguments
    args = parser.parse_args()
    if args.output_filename is None:
        joined_suffix = "joined" if args.join_hemispheres else "separated"
        diagonal_suffix = "_diagonal" if args.diagonal else ""
        args.output_filename = (
            f"rsa_subject_model_alignment_{joined_suffix}_{args.shift}{diagonal_suffix}"
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
    sessions = list(df_stim["session"].unique())
    total_rois = 180 if join_hemisphere else 360

    roi_session_rdms = []

    for roi in trange(
        1, total_rois + 1, desc="Preparing ROIs", position=2, leave=False
    ):
        roi_betas = get_subject_roi(
            subject, roi if not join_hemisphere else [roi, roi + 180]
        )
        session_rdms = []
        for session in sessions:
            session_subject_index = df_stim.query("session == @session")[
                "subject_index"
            ].values
            flat_rdm = compute_flat_rdm(roi_betas[session_subject_index])
            session_rdms.append(flat_rdm)

        roi_session_rdms.append(torch.stack(session_rdms))

    # Stack as a 3D tensor of (n_roi x n_session x n_flat_rdm_shape)
    features = torch.stack(roi_session_rdms)

    return features


def prepare_model_features(model_features: np.ndarray, df_stim):
    sessions = list(df_stim["session"].unique())
    n_layers = model_features.shape[1]

    layer_session_rdms = []

    for layer in trange(n_layers, desc="Preparing Layers", position=2, leave=False):
        session_rdms = []
        for session in sessions:
            session_nsd_ids = df_stim.query("session == @session")["nsd_id"].values
            session_features = model_features[session_nsd_ids, layer, :]
            flat_rdm = compute_flat_rdm(session_features)
            session_rdms.append(flat_rdm)

        layer_session_rdms.append(torch.stack(session_rdms))

    # Stack as a 3D tensor of (n_layers x n_sessions x n_flat_rdms)
    features = torch.stack(layer_session_rdms)

    return features


def compute_rsa_tensor(subject_features: torch.Tensor, model_features: torch.Tensor):
    """
    Computes RSA tensor with dimensions n_roi x n_layer x n_session.

    Args:
        subject_features: tensor of shape (n_roi, n_session, n_flat_rdm)
        model_features: tensor of shape (n_layer, n_session, n_flat_rdm)

    Returns:
        rsa_tensor: tensor of shape (n_roi, n_layer, n_session)
    """

    return torch.einsum("rsd,lsd->rls", subject_features, model_features).cpu().numpy()


def unravel_tensor(
    tensor: np.ndarray, subject: int, model_name: str, join_hemispheres: bool
) -> pd.DataFrame:
    # n_roi x n_layer x n_session.
    data = []
    n_roi, n_layer, n_session = tensor.shape
    for roi in range(n_roi):
        for layer in range(n_layer):
            for session in range(n_session):
                data.append(
                    {
                        "roi": roi + 1,  # ROI is indexed from 1
                        "layer": layer,  # Layer is indexed from 0
                        "session": session + 1,  # Session is indexed from 1
                        "similarity": tensor[roi, layer, session],
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
    df.session = df.session.astype("uint8")  # 1-40
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


def get_n_from_vector_size(vector_size: int) -> int:
    return int((1 + (1 + 8 * vector_size) ** 0.5) / 2)


def restrict_permutation(perm, n_common):
    return perm[perm < n_common]


@torch.jit.script
def optimized_permutation(
    perm: torch.Tensor,
    model_features: torch.Tensor,
    subject_features: torch.Tensor,
    triu_i: torch.Tensor,
    triu_j: torch.Tensor,
    inv_perm: torch.Tensor,
    order: torch.Tensor,
    arange_n: torch.Tensor,
    arange_N: torch.Tensor,
    n: int,
    N: int,
) -> torch.Tensor:
    # Reuse pre-allocated inv_perm
    inv_perm[perm] = arange_n

    new_i = inv_perm[triu_i]
    new_j = inv_perm[triu_j]

    lower = torch.minimum(new_i, new_j)
    upper = torch.maximum(new_i, new_j)

    tmp = (n - lower) * (n - lower - 1) // 2
    new_indices = (N - tmp) + (upper - lower - 1)

    # Reuse pre-allocated order
    order[new_indices] = arange_N

    result = torch.einsum("rsd,lsd->rls", subject_features, model_features[:, :, order])

    return result


def perform_permutations(
    subject_features: torch.Tensor,
    model_features: torch.Tensor,
    permutations: torch.Tensor,
    permutations_folder: Path,
    model_name: str,
    subject: int,
):
    n_permutations, maximal_n = permutations.shape
    n_roi, n_sessions, flat_n = (
        subject_features.shape
    )  # N equals (maximal_n * (maximal_n - 1)) // 2
    n_layers, n_sessions2, flat_n2 = model_features.shape

    assert n_sessions == n_sessions2, "Mismatch in number of sessions"
    assert flat_n == flat_n2, "Mismatch in number of features"

    n = get_n_from_vector_size(
        flat_n
    )  # Not neede cause we are using the same n for all subjects

    assert n == maximal_n, "Mismatch in number of stimuli"

    device = subject_features.device

    # The output is n_permutations x n_rois x n_layers x n_session
    rsa_permutations = torch.zeros(
        n_permutations, n_roi, n_layers, n_sessions, device=device
    )

    # Allocate these once outside the loop
    triu_i, triu_j = torch.triu_indices(n, n, offset=1, device=device)
    inv_perm = torch.empty(n, dtype=torch.long, device=device)
    order = torch.empty(flat_n, dtype=torch.long, device=device)
    arange_n = torch.arange(n, device=device)
    arange_N = torch.arange(flat_n, device=device)

    for i in trange(n_permutations, desc="Permutations", position=3, leave=False):
        permutations_i = permutations[i]
        if maximal_n > n:
            permutations_i = restrict_permutation(
                permutations_i, n
            )  # Not neede cause we are using the same n for all subjects

        rsa_permutations[i] = optimized_permutation(
            perm=permutations_i.to(device),
            model_features=model_features,
            subject_features=subject_features,
            triu_i=triu_i,
            triu_j=triu_j,
            inv_perm=inv_perm,
            order=order,
            arange_n=arange_n,
            arange_N=arange_N,
            n=n,
            N=flat_n,
        )

    rsa_permutations = rsa_permutations.cpu().numpy()

    filename_permutations = permutations_folder / f"rsa_permutations_subj-{subject}_{model_name}_{n_permutations}.npy"
    np.save(filename_permutations, rsa_permutations)


def compare_subject_model(
    model_features: np.ndarray,
    subject: int,
    join_hemisphere: bool,
    model_name: str,
    cache_folder: str,
    permutations: torch.Tensor = None,
    permutations_folder: Path = None,
):
    df_stim = get_resource("stimulus").query("subject == @subject and exists")
    subject_features = cached_prepare_subject_features(
        subject, df_stim, join_hemisphere, cache_folder
    )
    model_features = prepare_model_features(model_features, df_stim)

    if permutations is not None:
        perform_permutations(
            model_features=model_features,
            subject_features=subject_features,
            permutations=permutations,
            permutations_folder=permutations_folder,
            model_name=model_name,
            subject=subject,
        )

    # Compute RSA tensor
    rsa_tensor = compute_rsa_tensor(subject_features, model_features)

    return unravel_tensor(
        tensor=rsa_tensor,
        subject=subject,
        model_name=model_name,
        join_hemispheres=join_hemisphere,
    )


def get_permutations(perms, permutations_folder, max_shared_trials=750):
    if perms:
        permutations_folder.mkdir(exist_ok=True)
        filename_perms = (
            permutations_folder
            / f"subject_subject_permutations_{perms}_{max_shared_trials}.npy"
        )
        if not Path(filename_perms).exists():
            permutations = generate_batched_permutations(
                n_perm=perms, n=max_shared_trials, device="cpu"
            )
            # Cache them as a npy file
            np.save(filename_perms, permutations.numpy())
            del permutations
            gc.collect()
            print("Generated permutations in", filename_perms)

        permutations = torch.tensor(np.load(filename_perms), device="cpu")
    else:
        permutations = None

    return permutations


@alert
def main():

    args = parse_args()
    output_filename = Path(args.output_filename)
    models = load_model_paths()
    join_hemisphere = args.join_hemispheres
    cache_folder = args.cache_folder
    n_subjects = 8
    permutations_folder = Path(args.permutations_folder)
    perms = args.n_permutations
    permutations = get_permutations(perms, permutations_folder)

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
                permutations=permutations,
                permutations_folder=permutations_folder,
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

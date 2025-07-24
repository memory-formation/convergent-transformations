"""
rsa_nsd_subject_model_alignment_controled.py

Compute RSA alignment between subject-specific fMRI RDMs and deep model representations
across cortical ROIs in the NSD dataset, with optional control for confounding variables
(e.g., image statistics, semantics). 

This script evaluates the similarity between each subject's neural representations and
model layer representations using controlled partial correlation (e.g., semi-partial RSA).
The analysis is repeated across models, layers, subjects, and control matrices.

Key Features:
- Computes flat RSA (correlation of RDMs) across subject ROIs x model layers.
- Supports multiple control regressors (from `.parquet` control matrices).
- Computes both raw (uncontrolled) and controlled similarity scores.
- Supports 180 (joined hemispheres) or 360 (separate) ROI configurations.
- Efficient GPU-based computation via PyTorch.

Usage:
    python rsa_nsd_subject_model_alignment_controled.py [--join_hemispheres] [--output_filename out.parquet]

Arguments:
    --output_filename         Output filename (default is auto-generated based on flags).
    --join_hemispheres        Use 180 ROIs by merging left/right hemispheres (default: False).

Expected Inputs:
- NSD beta responses and ROI masks (via `convergence` access functions).
- Model feature files as `.pt` tensors: shape [n_stimuli x n_layers x n_features].
- Control RDMs stored in `control_matrices/control_*.parquet`.

Output:
- One `.parquet` file per model, with RSA values per subject, ROI, model layer, and control condition.
  Columns include: `subject`, `roi`, `layer`, `similarity`, `model`, `control`, `distance`.

Requirements:
- Model features stored in: `/mnt/tecla/Results/convergence/features/nsd/all/*.pt`
- Control matrices in: `control_matrices/control_*.parquet`
- Requires GPU (`cuda`) and access to NSD dataset via `get_subject_roi()` and `get_resource()`.

Example:
    python rsa_nsd_subject_model_alignment_controled.py --join_hemispheres

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
TEST = False
CACHE_SUBJECTS = True

model_cache = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cross-subject similarities with various metrics and configurations."
    )

    # Argument for output filename
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Output filename for the results. Default is based on the shift value.",
    )
    parser.add_argument(
        "--join_hemispheres",
        action="store_true",
    )
    # Parse the arguments
    args = parser.parse_args()
    if args.output_filename is None:
        joined_suffix = "joined" if args.join_hemispheres else "separated"
        test_suffix = "_test" if TEST else ""
        args.output_filename = f"rsa_subject_model_alignment_controled_{joined_suffix}{test_suffix}.parquet"
    if not args.output_filename.endswith(".parquet"):
        args.output_filename += ".parquet"

    return args


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


def prepare_subject_features(subject, subject_indexes, join_hemisphere: bool):

    total_rois = 180 if join_hemisphere else 360
    if TEST:
        total_rois = 4  # TEST

    roi_session_rdms = []

    for roi in trange(
        1, total_rois + 1, desc="Preparing ROIs", position=2, leave=False
    ):
        roi_betas = get_subject_roi(
            subject, roi if not join_hemisphere else [roi, roi + 180]
        )

        flat_rdm = compute_flat_rdm(roi_betas[subject_indexes])
        roi_session_rdms.append(flat_rdm)

    # Stack as a 2D tensor of (n_roi x n_flat_rdm_shape)
    features = torch.stack(roi_session_rdms)

    return features


@torch.jit.script
def einsum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,ij->i", a, b)


@torch.jit.script
def matprod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b.T)


def compute_rsa_tensor(
    subject_features: torch.Tensor,
    model_features: torch.Tensor,
):
    """
    Computes RSA tensor with dimensions n_roi x n_roi.

    Args:
        subject_features: tensor of shape (n_roi, n_session, n_flat_rdm)
        model_features: tensor of shape (n_layer, n_session, n_flat_rdm)

    Returns:
        rsa_tensor: tensor of shape (n_roi, n_layer, n_session)
    """

    rsa_matrix = matprod(subject_features, model_features)

    return rsa_matrix.cpu().numpy()


def unravel_tensor(
    tensor: np.ndarray,
    subject: int,
    model_name: str,
    join_hemispheres: bool,
    control: str,
    distance: str,
) -> pd.DataFrame:
    # If not diagonal it receives a matrix [n_roi, n_layers]
    data = []

    for roi in range(tensor.shape[0]):
        for layer in range(tensor.shape[1]):
                data.append(
                    {
                        "roi": roi + 1,  # ROI is indexed from 1
                        "layer": layer,  # layers is indexed from 0
                        "similarity": float(tensor[roi, layer]),
                    }
                )

    df = pd.DataFrame(data)
    df.similarity = df.similarity.astype("float32")
    df["subject"] = subject
    df["subject"] = df["subject"].astype("uint8")
    df["model"] = model_name
    df["model"] = df["model"].astype("category")
    df["join_hemispheres"] = join_hemispheres
    df["join_hemispheres"] = df["join_hemispheres"].astype("bool")
    df["control"] = control
    df["control"] = df["control"].astype("category")
    df["distance"] = distance
    df["distance"] = df["distance"].astype("category")
    df["roi"] = df["roi"].astype("uint16")
    df["layer"] = df["layer"].astype("uint8")

    return df


def pandas_to_rdm(df, nsd_ids=None, distance="euclidean"):
    """
    Convert a pandas DataFrame to a square distance matrix.
    """
    if nsd_ids is None:
        nsd_ids = df.nsd_id.values

    df_indexed = df.set_index("nsd_id")
    df_indexed = df_indexed.loc[nsd_ids]
    values = df_indexed.values.astype(np.float32)

    if distance == "euclidean":
        rdm = np.sqrt(np.sum((values[:, None] - values[None, :]) ** 2, axis=-1))

    elif distance == "pearson":  # Not valid for 1D data
        rdm = 1 - np.corrcoef(values)

    return rdm


def control_pandas_to_rdm(df, nsd_ids=None, distance="euclidean"):
    """
    Convert a pandas DataFrame to a square distance matrix.
    """
    if nsd_ids is None:
        nsd_ids = df.nsd_id.values

    df_indexed = df.set_index("nsd_id")
    df_indexed = df_indexed.loc[nsd_ids]
    values = df_indexed.values.astype(np.float32)

    if distance == "euclidean":
        rdm = np.sqrt(np.sum((values[:, None] - values[None, :]) ** 2, axis=-1))

    elif distance == "pearson":  # Not valid for 1D data
        rdm = 1 - np.corrcoef(values)

    # To torch
    rdm = torch.tensor(rdm, device=DEVICE, dtype=torch.float32)

    return create_flat_normalize_rdm(rdm)


def compute_controled_rsa(
    subject_features: torch.Tensor,  # [n_roi, n_flat_rdm_shape]
    model_features: torch.Tensor,  # [n_roi, n_flat_rdm_shape]
    df_control: pd.DataFrame,
    nsd_ids,
    distance="euclidean",
) -> pd.DataFrame:

    rdm_control = control_pandas_to_rdm(df_control, nsd_ids=nsd_ids, distance=distance)


    r_AC = torch.einsum("ij,j->i", subject_features, rdm_control)
    r_BC = torch.einsum("ij,j->i", model_features, rdm_control)


    r_AB = torch.mm(subject_features, model_features.T)

    partial_r = (r_AB - r_AC[:, None] * r_BC[None, :]) / torch.sqrt(
            (1 - r_AC[:, None] ** 2) * (1 - r_BC[None, :] ** 2)
    )

    return partial_r

def get_subject_indexes(subject, restricted_nsd_ids):
    df_stim = get_resource("stimulus")
    df_stim = df_stim[df_stim.subject == subject]
    df_stim = df_stim.query("nsd_id in @restricted_nsd_ids and shared and exists")
    df_stim = df_stim[["nsd_id", "subject_index", "repetition"]].copy()

    return df_stim

def prepare_model_features(model_features: np.ndarray, nsd_ids):
    n_layers = model_features.shape[1]
    if TEST:
        n_layers = 3

    layer_rdms = []
    model_subset = model_features[nsd_ids, :, :]

    for layer in trange(n_layers, desc="Preparing Layers", position=2, leave=False):
        layer_features = model_subset[:, layer, :]
        flat_rdm = compute_flat_rdm(layer_features)
        layer_rdms.append(flat_rdm)


    # Stack as a 2D tensor of (n_layers x n_flat_rdms)
    features = torch.stack(layer_rdms)

    return features


def compare_subject_subject(
    subject: int,
    model_features: torch.Tensor,
    model_name: str,
    join_hemisphere: bool,
    control_files: list[Path],
    control_nsd_ids: list[int],
    control_distance: str = "euclidean",
):

    df_subject = get_subject_indexes(subject, restricted_nsd_ids=control_nsd_ids)

    subject_indexes = df_subject.subject_index.values

    if CACHE_SUBJECTS and subject in model_cache:
        subject_features = model_cache[subject]
    else:
        subject_features = prepare_subject_features(
            subject=subject,
            subject_indexes=subject_indexes,
            join_hemisphere=join_hemisphere,
        )
        if CACHE_SUBJECTS:
            model_cache[subject] = subject_features
            
    nsd_ids = df_subject.nsd_id.values
    model_features = prepare_model_features(
        model_features=model_features,
        nsd_ids=nsd_ids,
    )

    # Compute the uncontrolled RSA tensor
    rsa_tensor = compute_rsa_tensor(
        subject_features, model_features,
    )

    
    df_uncontrolled = unravel_tensor(
        tensor=rsa_tensor,
        subject=subject,
        model_name=model_name,
        join_hemispheres=join_hemisphere,
        control="uncontrolled",
        distance="uncontrolled",
    )
    dfs = [df_uncontrolled]

    # Control RDM
    for control_file in tqdm(control_files, desc="Controls", position=3, leave=False):
        df_control = pd.read_parquet(control_file)
        control_name = control_file.stem.replace("control_", "")
        rsa_tensor_controled = compute_controled_rsa(
            subject_features=subject_features,
            model_features=model_features,
            df_control=df_control,
            nsd_ids=nsd_ids,
            distance=control_distance,
        )
        df_controled = unravel_tensor(
            tensor=rsa_tensor_controled,
            subject=subject,
            model_name=model_name,
            join_hemispheres=join_hemisphere,
            control=control_name,
            distance=control_distance,
        )
        dfs.append(df_controled)


    dfs = pd.concat(dfs, ignore_index=True)
    dfs["control"] = dfs["control"].astype("category")
    dfs["distance"] = dfs["distance"].astype("category")
    return dfs


def load_model_paths():
    models_folder = Path("/mnt/tecla/Results/convergence/features/nsd/all")
    models = list(models_folder.glob("*.pt"))
    models = [model for model in models if not model.stem.endswith("-pixtral")]
    return models


@alert(disable=TEST)
def main():
    args = parse_args()
    output_filename = Path(args.output_filename)
    join_hemisphere = args.join_hemispheres
    n_subjects = get_resource("stimulus").subject.nunique()
    models = load_model_paths()
    if TEST:
        print("TESTING")
        n_subjects = 1  # TEST
        models = models[:1]


    # Control files
    control_folder = Path("control_matrices")
    control_files = list(control_folder.glob("control_*.parquet"))
    # Assume all control files have the same nsd_ids
    restricted_nsd_ids = pd.read_parquet(control_files[0]).nsd_id.unique()

    for model_path in tqdm(models, position=0, desc="Model", leave=False):
        results = []
        model_name = model_path.stem
        filename_model = Path(f"{model_name}_{output_filename.name}")
        if filename_model.exists() and not TEST:
            print("Skipping model", model_name)
            continue

        model_features = load_model_features(model_path)

        send_alert(f"Processing model {model_name}")

        for subject in trange(
            1, n_subjects + 1, position=1, desc="Subj-j", leave=False
        ):
            df_model_subject = compare_subject_subject(
                subject=subject,
                model_features=model_features,
                model_name=model_name,
                join_hemisphere=join_hemisphere,
                control_files=control_files,
                control_nsd_ids=restricted_nsd_ids,
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

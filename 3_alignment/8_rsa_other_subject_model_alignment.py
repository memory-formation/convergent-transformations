"""
rsa_other_subject_model_alignment.py

Compute RSA between brain ROIs and deep model layers for individual subjects using either the
THINGS or BOLD5000 datasets. Supports hemisphere-joined comparisons and subset selection within
BOLD5000 (COCO, Scene, or ImageNet). Outputs a multi-session, ROIxLayer similarity matrix.

---

**Key Features:**
- Computes RSA between subject-specific RDMs (per ROI) and deep neural model RDMs (per layer).
- Supports both THINGS and BOLD5000 datasets (and its COCO/Scene/ImageNet subsets).
- Handles session-wise data: RSA is computed separately for each scan session.
- Allows hemisphere merging (`--join_hemispheres`) to collapse 360 ROIs into 180.
- Uses efficient GPU-based RDM computation and normalization.
- Modular loading of model feature tensors (.pt format with `["feats"]`).

---

**Arguments:**
    --output_filename         Output filename (.parquet). Auto-generated if not specified.
    --join_hemispheres        Whether to merge left/right hemispheres. Default: False (360 ROIs).
    --dataset                 Dataset to use: ["things", "bold5000", "bold5000-coco", ...].

---

**Inputs:**
- fMRI subject betas: Loaded via internal `get_betas_roi_*` functions.
- Stimulus metadata: Trial indices per subject and session from `stimulus.csv`.
- Model features: Precomputed RDM-friendly tensors, loaded from `.pt` files per dataset.

---

**Outputs:**
- `.parquet` file with per-subject RSA results:
    - `subject`: subject ID
    - `roi`: ROI index (1-360 or 1-180)
    - `layer`: model layer index
    - `session`: scan session index (1-n)
    - `similarity`: RSA score (Pearson correlation by default)
    - `model`: model name (from file)
    - `dataset`: dataset name (e.g., "things", "bold5000-coco")
    - `join_hemispheres`: whether hemispheres were merged

---

**Workflow Summary:**
1. For each model:
    - Load model features: shape (n_images x n_layers x d).
2. For each subject:
    - Load stimulus trials and betas.
    - Compute ROI-wise RDMs per session.
    - Compute model-layer RDMs per session.
    - Align and compare using cosine distance (1 - dot product of L2-normalized vectors).
3. Save all results in a tidy `.parquet` file.

---

**Example Usage:**
```bash
python rsa_other_subject_model_alignment.py \
    --dataset bold5000-coco \
    --join_hemispheres \
    --output_filename rsa_bold5000_coco_joined.parquet
```
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
from convergence.things import (
    get_betas_roi as get_betas_roi_things,
    get_resource as get_resource_things,
)
from convergence.bold5000 import (
    get_betas_roi as get_subject_roi_bold5000,
    get_resource as get_resource_bold5000,
)


DEVICE = "cuda"
TEST = False

FEATURES_FOLDERS = {
    "things": "/mnt/tecla/Results/convergence/features/things/all",
    "bold5000": "/mnt/tecla/Results/convergence/features/bold5000/all",
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Intersubject for THINGS and BOLD5000 datasets"
    )
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument("--join_hemispheres", action="store_true")
    dataset = [
        "things",
        "bold5000",
        "bold5000-coco",
        "bold5000-scenes",
        "bold5000-imagenet",
    ]
    parser.add_argument("--dataset", type=str, default="things", choices=dataset)
    args = parser.parse_args()

    joined_suffix = "joined" if args.join_hemispheres else "separated"
    dataset = args.dataset
    test_suffix = "_test" if TEST else ""
    default_filename = (
        f"rsa_{dataset}_subject_model_alignment_{joined_suffix}{test_suffix}.parquet"
    )

    if args.output_filename is None:
        args.output_filename = default_filename

    return args


def load_model_paths(dataset):
    models_folder = Path(FEATURES_FOLDERS[dataset])
    models = list(models_folder.glob("*.pt"))
    return models


def load_model_features(model_path: Path):
    features = torch.load(model_path, weights_only=True)
    features = (
        features["feats"].to(torch.float32).numpy()
    )  # n_stim x n_layers x n_features
    return features


def get_dataset_stimuli(dataset):
    """Get dataset stimuli"""
    if len(dataset.split("-")) > 1:
        dataset, subset = dataset.split("-")
    else:
        subset = None
    if dataset == "bold5000":
        df = get_resource_bold5000("stimulus")
        if subset:
            df = df.query(f"stim_source == '{subset}'").reset_index(drop=True).copy()
        return df
    elif dataset == "things":
        return get_resource_things("stimulus").rename(columns={"subject_id": "subject"})
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_subject_roi(subject, roi, dataset, join_hemisphere=False):
    """Get subject ROI betas"""
    if "-" in dataset:
        dataset = dataset.split("-")[0]
    roi = roi if not join_hemisphere else [roi, roi + 180]
    """Get subject ROI betas"""
    if dataset == "bold5000":
        return get_subject_roi_bold5000(subject, roi)
    elif dataset == "things":
        return get_betas_roi_things(subject, roi)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_flat_normalize_rdm(
    rdm: torch.Tensor, triu_indices: torch.Tensor = None
) -> torch.Tensor:
    """Generates a flat RDM from a square RDM and normalizes it."""
    if triu_indices is None:
        triu_indices = torch.triu_indices(rdm.size(0), rdm.size(0), offset=1)
    rdm_flat = rdm[triu_indices[0], triu_indices[1]]
    rdm_flat = rdm_flat - rdm_flat.mean()
    rdm_flat /= rdm_flat.norm()
    return rdm_flat


def compute_flat_rdm(betas_subset: np.ndarray, q=0.003):
    """Computes a flat RDM from a set of betas."""
    a, b = np.quantile(betas_subset, [q, 1 - q])
    betas_subset = np.clip(betas_subset, a, b)

    betas_subset = (betas_subset - a) / (b - a)
    betas_subset = torch.tensor(betas_subset, device=DEVICE, dtype=torch.float32)

    # Compute RDM
    betas_subset = betas_subset - betas_subset.mean(dim=1, keepdim=True)
    betas_subset = torch.nn.functional.normalize(betas_subset, dim=1)
    rdm = 1 - torch.mm(betas_subset, betas_subset.t())
    return create_flat_normalize_rdm(rdm)


def prepare_subject_features(
    subject: int,
    df_stimuli: pd.DataFrame,
    dataset: str,
    join_hemisphere: bool = False,
    subject_indexes_column: str = "subject_index",
    session_column: str = "session",
):
    """Prepares the features for a given subject."""

    total_rois = 180 if join_hemisphere else 360
    if TEST:
        total_rois = 4

    sessions = df_stimuli[session_column].unique()

    features = (
        {}
    )  # Not all sessions have the same number of stimuli. Cannot stack as single tensor
    for session in sessions:
        features[session] = []

    for roi in trange(
        1, total_rois + 1, desc="Preparing ROIs", position=2, leave=False
    ):
        roi_betas = get_subject_roi(
            subject=subject, roi=roi, dataset=dataset, join_hemisphere=join_hemisphere
        )
        for session in sessions:
            df_session = df_stimuli.query(f"{session_column} == {session}")
            subject_indexes = df_session[subject_indexes_column].values
            flat_rdm = compute_flat_rdm(roi_betas[subject_indexes])
            features[session].append(flat_rdm)

    # Stack each session as 2D tensor of (n_roi x n_flat_rdm_shape)
    for session in sessions:
        features[session] = torch.stack(features[session], dim=0).to(DEVICE)

    return features


def compute_rsa_tensor(
    rdms_subj: dict[torch.Tensor],
    rdms_model: dict[torch.Tensor],
):
    """
    Computes RSA tensor with dimensions n_sessions x n_roi x n_layers.

    Args:
        rdms_subj: tensor of shape n_session x (n_roi, n_flat_rdm)
        rdms_model: tensor of shape n_session x (n_layers, n_flat_rdm)

    Returns:
        rsa_tensor: tensor of shape n_session x (n_roi, n_layers)
    """
    sessions = list(rdms_subj.keys())
    sessions.sort()  # Not neede: Ensure sorted order

    rsa_tensor = []
    for session in sessions:
        rdm_i = rdms_subj[session]  # shape (n_roi, n_flat_rdm)
        rdm_j = rdms_model[session]  # shape (n_roi, n_flat_rdm)
        session_rsa_tensor = torch.mm(rdm_i, rdm_j.T)  # shape (n_roi, n_layer)
        session_rsa_tensor = session_rsa_tensor.to("cpu").numpy()

        rsa_tensor.append(session_rsa_tensor)


    # Stack each session as 3D tensor of (n_session x n_roi x n_layers)
    rsa_tensor = np.stack(rsa_tensor, axis=0)  # shape (n_session, n_roi, n_layers)
    return rsa_tensor


def prepare_model_features(
    model_features: np.ndarray,
    df_stimuli: pd.DataFrame,
    image_column: str = "bold_position",
    session_column: str = "session",
):
    n_layers = model_features.shape[1]
    if TEST:
        n_layers = 3

    sessions = df_stimuli[session_column].unique()
    sessions.sort()  # Not needed: Ensure sorted order

    features = {}

    for session in sessions:
        layer_rdms = []
        df_session = df_stimuli.query(f"{session_column} == {session}")
        stimulus_ids = df_session[image_column].values
        model_subset = model_features[stimulus_ids, :, :]

        for layer in trange(n_layers, desc="Preparing Layers", position=2, leave=False):
            layer_features = model_subset[:, layer, :]
            flat_rdm = compute_flat_rdm(layer_features)
            layer_rdms.append(flat_rdm)

        # Stack as a 2D tensor of (n_layers x n_flat_rdms)
        features[session] = torch.stack(layer_rdms)

    return features


def unravel_tensor(
    tensor: np.ndarray,
    subject: int,
    model_name: str,
    join_hemispheres: bool,
    dataset: str,
) -> pd.DataFrame:
    """Unravel the tensor into a DataFrame."""
    data = []
    for session in range(tensor.shape[0]):
        for roi in range(tensor.shape[1]):
            for layer in range(tensor.shape[2]):
                data.append(
                    {
                        "session": session + 1,  # session is indexed from 1
                        "roi": roi + 1,  # ROI is indexed from 1
                        "layer": layer + 1,  # ROI is indexed from 1
                        "similarity": float(tensor[session, roi, layer]),
                    }
                )

    df = pd.DataFrame(data)
    df["similarity"] = df["similarity"].astype("float32")
    df["roi"] = df["roi"].astype("uint16")
    df["layer"] = df["layer"].astype("uint16")
    df["subject"] = subject
    df["subject"] = df["subject"].astype("uint8")
    df["model"] = model_name
    df["model"] = df["model"].astype("category")
    df["session"] = df["session"].astype("uint8")
    df["join_hemispheres"] = join_hemispheres
    df["join_hemispheres"] = df["join_hemispheres"].astype("bool")
    df["dataset"] = dataset
    df["dataset"] = df["dataset"].astype("string").astype("category")

    return df


def compare_subject_subject(
    subject: int,
    model_features: np.ndarray,
    model_name: str,
    join_hemisphere: bool,
    dataset: str,
):
    df_stimuli = get_dataset_stimuli(dataset).query("subject == @subject")

    subject_features = prepare_subject_features(
        subject=subject,
        df_stimuli=df_stimuli,
        dataset=dataset,
        join_hemisphere=join_hemisphere,
    )
    image_column = "bold_position" if "bold5000" in dataset else "things_id"
    model_features = prepare_model_features(
        model_features=model_features,
        df_stimuli=df_stimuli,
        image_column=image_column,
    )
    rsa_tensor = compute_rsa_tensor(subject_features, model_features)

    return unravel_tensor(
        tensor=rsa_tensor,
        subject=subject,
        model_name=model_name,
        join_hemispheres=join_hemisphere,
        dataset=dataset,
    )


@alert(disable=TEST, output=True)
def main():
    args = parse_args()
    output_filename = Path(args.output_filename)
    join_hemisphere = args.join_hemispheres
    dataset = args.dataset
    df_stimuli = get_dataset_stimuli(dataset)
    n_subjects = df_stimuli.subject.nunique()
    models = load_model_paths(dataset.split("-")[0])
    if TEST:
        n_subjects = 2
        models = models[:1]

    results = []
    for model in tqdm(models, desc="Models", position=0, leave=False):
        model_features = load_model_features(model)
        model_name = model.stem
        if not TEST:
            send_alert(f"Processing model {model_name}")
        for subject in trange(
            1, n_subjects + 1, position=1, desc="Subject", leave=False
        ):

            df_subject_model = compare_subject_subject(
                subject=subject,
                model_features=model_features,
                model_name=model_name,
                join_hemisphere=join_hemisphere,
                dataset=dataset,
            )
            results.append(df_subject_model)
            gc.collect()
            torch.cuda.empty_cache()


    results = pd.concat(results, ignore_index=True)
    results.to_parquet(output_filename, index=False)
    return output_filename


if __name__ == "__main__":
    main()

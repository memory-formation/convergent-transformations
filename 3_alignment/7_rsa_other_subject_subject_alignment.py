"""
rsa_other_subject_subject_alignment.py

Compute pairwise inter-subject RSA alignment across brain ROIs for the THINGS or BOLD5000 datasets.

This script compares the representational dissimilarity matrices (RDMs) derived from the brain
activity of two different subjects, ROI-wise and session-wise, computing a similarity matrix 
between ROIs from each subject. The analysis can be applied to the full datasets or to BOLD5000 
subsets (COCO, Scene, ImageNet), and optionally merges hemispheres.

---

**Key Features:**
- Computes inter-subject ROI-to-ROI alignment using representational similarity (cosine).
- Supports BOLD5000 (and subsets) and THINGS datasets.
- Allows hemisphere merging (`--join_hemispheres`) to reduce from 360 to 180 ROIs.
- Matches repeated image presentations across subjects using a shared trial index.
- Returns RSA matrices for each subject pair and session.

---

**Arguments:**
    --output_filename         Path to save the final DataFrame (.parquet).
    --join_hemispheres        Merge left/right ROIs (360 → 180). Default: False.
    --dataset                 Dataset to use: one of ["bold5000", "things", "bold5000-coco", ...].

---

**Outputs:**
- `.parquet` file with long-format DataFrame containing:
    - `subject_i`, `subject_j`: subject pair
    - `roi_x`, `roi_y`: ROI pair indices
    - `session`: scan session number
    - `similarity`: RSA score (cosine similarity of flattened RDMs)
    - `join_hemispheres`, `dataset`

---

**Workflow Summary:**
1. For each subject pair (excluding self-pairs):
    - Find common repeated trials (by image and repetition ID).
    - For each ROI in each subject:
        - Extract beta responses for shared trials.
        - Compute cosine-normalized RDMs.
    - Compare RDMs from subject_i and subject_j across sessions.
2. Save the result tensor (n_sessions x n_rois x n_rois) into a tidy DataFrame.

---

**Example Usage:**
```bash
python rsa_other_subject_subject_alignment.py \
    --dataset bold5000-scenes \
    --join_hemispheres \
    --output_filename rsa_bold5000_scenes_subject_subject_joined.parquet
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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Intersubject for THINGS and BOLD5000 datasets"
    )
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument("--join_hemispheres", action="store_true")
    dataset = ["bold5000", "things", "bold5000-coco", "bold5000-scenes", "bold5000-imagenet"]
    parser.add_argument("--dataset", type=str, default="bold5000", choices=dataset)
    args = parser.parse_args()

    joined_suffix = "joined" if args.join_hemispheres else "separated"
    dataset = args.dataset
    test_suffix = "_test" if TEST else ""
    default_filename = (
        f"rsa_{dataset}_subject_subject_alignment_{joined_suffix}{test_suffix}.parquet"
    )

    if args.output_filename is None:
        args.output_filename = default_filename

    return args


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


def get_common_indexes(subject_i: int, subject_j: int, dataset: str) -> pd.DataFrame:
    """Get common indexes between two subjects for a given dataset"""
    df_stimuli = get_dataset_stimuli(dataset)
    image_key = "bold_position" if "bold5000" in dataset else "things_id"

    columns = [image_key, "subject", "session", "subject_index", "repetition"]

    df_stimuli = df_stimuli[columns]
    df_i = df_stimuli.query(f"subject == {subject_i}").copy()
    df_j = df_stimuli.query(f"subject == {subject_j}").copy()

    df_merge = df_i.merge(df_j, on=[image_key, "repetition"], suffixes=("_i", "_j"))

    return df_merge


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
    df_merge: pd.DataFrame,
    subject_indexes_column: str,
    join_hemisphere: bool,
    dataset: str,
    session_column: str = "session_i",
):
    """Prepares the features for a given subject."""

    total_rois = 180 if join_hemisphere else 360
    if TEST:
        total_rois = 3

    sessions = df_merge[session_column].unique()

    features = {}
    for session in sessions:
        features[session] = []

    for roi in trange(
        1, total_rois + 1, desc="Preparing ROIs", position=2, leave=False
    ):
        roi_betas = get_subject_roi(
            subject=subject, roi=roi, dataset=dataset, join_hemisphere=join_hemisphere
        )
        for session in sessions:
            df_session = df_merge.query(f"{session_column} == {session}")
            subject_indexes = df_session[subject_indexes_column].values
            flat_rdm = compute_flat_rdm(roi_betas[subject_indexes])
            features[session].append(flat_rdm)

    # Stack each session as 2D tensor of (n_roi x n_flat_rdm_shape)
    for session in sessions:
        features[session] = torch.stack(features[session], dim=0).to(DEVICE)

    return features


def compute_rsa_tensor(
    rdms_subj_i: dict[torch.Tensor],
    rdms_subj_j: dict[torch.Tensor],
):
    """
    Computes RSA tensor with dimensions n_roi x n_roi.

    Args:
        rdms_subj_i: tensor of shape n_session x (n_roi, n_flat_rdm)
        rdms_subj_j: tensor of shape n_session x (n_roi, n_flat_rdm)

    Returns:
        rsa_tensor: tensor of shape n_session x (n_roi, n_roi)
    """
    sessions = list(rdms_subj_i.keys())
    sessions.sort()  # Not neede: Ensure sorted order

    rsa_tensor = []
    for session in sessions:
        rdm_i = rdms_subj_i[session]  # shape (n_roi, n_flat_rdm)
        rdm_j = rdms_subj_j[session]  # shape (n_roi, n_flat_rdm)
        session_rsa_tensor = torch.mm(rdm_i, rdm_j.T)  # shape (n_roi(i), n_roi(j))
        session_rsa_tensor = session_rsa_tensor.to("cpu").numpy()
        rsa_tensor.append(session_rsa_tensor)

    # Stack each session as 3D tensor of (n_session x n_roi(i) x n_roi(j))
    rsa_tensor = np.stack(rsa_tensor, axis=0)  # shape (n_session, n_roi(i), n_roi(j))
    return rsa_tensor


def unravel_tensor(
    tensor: np.ndarray,
    subject_i: int,
    subject_j: int,
    join_hemispheres: bool,
    dataset: str,
) -> pd.DataFrame:
    """Unravel the tensor into a DataFrame."""
    data = []
    for session in range(tensor.shape[0]):
        for roi_x in range(tensor.shape[1]):
            for roi_y in range(tensor.shape[2]):
                data.append(
                    {
                        "session": session + 1,  # session is indexed from 1
                        "roi_x": roi_x + 1,  # ROI is indexed from 1
                        "roi_y": roi_y + 1,  # ROI is indexed from 1
                        "similarity": float(tensor[session, roi_x, roi_y]),
                    }
                )

    df = pd.DataFrame(data)
    df["similarity"] = df["similarity"].astype("float32")
    df["roi_x"] = df["roi_x"].astype("uint16")
    df["roi_y"] = df["roi_y"].astype("uint16")
    df["subject_i"] = subject_i
    df["subject_i"] = df["subject_i"].astype("uint8")
    df["subject_j"] = subject_j
    df["subject_j"] = df["subject_j"].astype("uint8")
    df["join_hemispheres"] = join_hemispheres
    df["join_hemispheres"] = df["join_hemispheres"].astype("bool")
    df["dataset"] = dataset
    df["rep_shift"] = df["dataset"].astype("string").astype("category")

    return df


def compare_subject_subject(
    subject_i: int,
    subject_j: int,
    join_hemisphere: bool,
    dataset: str,
):
    df_merge = get_common_indexes(
        subject_i=subject_i, subject_j=subject_j, dataset=dataset
    )

    subject_features_i = prepare_subject_features(
        subject=subject_i,
        df_merge=df_merge,
        dataset=dataset,
        subject_indexes_column="subject_index_i",
        join_hemisphere=join_hemisphere,
    )
    subject_features_j = prepare_subject_features(
        subject=subject_j,
        df_merge=df_merge,
        dataset=dataset,
        subject_indexes_column="subject_index_j",
        join_hemisphere=join_hemisphere,
    )

    rsa_tensor = compute_rsa_tensor(subject_features_i, subject_features_j)
    
    return unravel_tensor(
        tensor=rsa_tensor,
        subject_i=subject_i,
        subject_j=subject_j,
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
    if TEST:
        n_subjects = 2

    results = []
    for subject_i in trange(1, n_subjects + 1, position=0, desc="Subj-i", leave=False):
        send_alert(f"Processing subject {subject_i}")
        for subject_j in trange(
            1, n_subjects + 1, position=1, desc="Subj-j", leave=False
        ):
            if subject_i == subject_j:
                continue
            df_subject = compare_subject_subject(
                subject_i=subject_i,
                subject_j=subject_j,
                join_hemisphere=join_hemisphere,
                dataset=dataset,
            )
            results.append(df_subject)
            gc.collect()
            torch.cuda.empty_cache()


    results = pd.concat(results, ignore_index=True)
    results.to_parquet(output_filename, index=False)
    return output_filename


if __name__ == "__main__":
    main()

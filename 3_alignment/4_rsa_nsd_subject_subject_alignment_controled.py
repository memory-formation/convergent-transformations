"""
rsa_nsd_subject_subject_alignment_controled.py

Compute representational similarity (RSA) between subject pairs in the NSD dataset,
optionally controlling for confounding representational sources (e.g., semantic embeddings, pixel similarity)
and shifting repetition indices across subjects.

This script performs RSA across pairs of subjects at the ROI level by computing RDMs on shared image sets, 
optionally:
- Joining left and right hemisphere ROIs
- Using only the diagonal (within-ROI) similarities
- Shifting trial repetitions across subjects to enforce cross-session alignment
- Controlling for representational confounds using partial correlation with external RDMs

Main features:
- Supports full or partial RSA (within-ROI only)
- Supports partial RSA using any number of external control RDMs
- Efficient GPU-based computation and batch memory management
- Caches intermediate files and supports TEST mode for debugging

Arguments:
    --output_filename           Output file prefix (auto-named if not provided)
    --join_hemispheres          Merge hemispheres into 180 ROIs (default: False = 360 ROIs)
    --shift                     Shift in repetition index to align trials across subjects (default: 1)
    --diagonal                  Use only diagonal ROIxROI comparisons (default: False)

Inputs:
- NSD betas via `get_subject_roi()` from `convergence.nsd`
- Stimulus metadata from `get_resource("stimulus")`
- Control RDMs in `control_matrices/control_*.parquet`
    Each file should contain trial-level features with a column `nsd_id`

Output:
- A `.parquet` file per subject_i:
    Columns:
        ['roi', 'subject_i', 'subject_j', 'similarity', 'rep_shift', 'control', 'distance']
    or (if not diagonal):
        ['roi_x', 'roi_y', 'subject_i', 'subject_j', 'similarity', 'rep_shift', 'control', 'distance']

Usage Example:
    python rsa_nsd_subject_subject_alignment_controled.py --join_hemispheres --shift 1 --diagonal

Test Mode:
- Set `TEST = True` to limit to 1 subject and 4 ROIs for debugging purposes

Notes:
- Control RDMs are used in a partial correlation framework to isolate variance unique to subject representations
- The script assumes all control RDMs use the same set of NSD trial IDs
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
    parser.add_argument(
        "--shift",
        type=int,
        default=1,
        help="Shift the repetitions for the second subject.",
    )
    parser.add_argument(
        "--diagonal",
        action="store_true",
    )
    # Parse the arguments
    args = parser.parse_args()
    if args.output_filename is None:
        joined_suffix = "joined" if args.join_hemispheres else "separated"
        diagonal_suffix = "_diagonal" if args.diagonal else ""
        test_suffix = "_test" if TEST else ""
        args.output_filename = f"rsa_subject_subject_alignment_controled_{joined_suffix}_{args.shift}{diagonal_suffix}{test_suffix}.parquet"
    if not args.output_filename.endswith(".parquet"):
        args.output_filename += ".parquet"

    return args


def get_common_indexes(subject_i, subject_j, shift=0, restricted_nsd_ids=None):

    df = get_resource("stimulus")
    if restricted_nsd_ids is not None:
        df = df.query(f"nsd_id in @restricted_nsd_ids")

    df_i = df.query(f"subject == {subject_i} and shared and exists")
    df_j = df.query(f"subject == {subject_j} and shared and exists")

    df_i = df_i[["subject", "nsd_id", "subject_index", "repetition"]]
    df_i = df_i.rename(
        columns={
            "subject": "subject_i",
            "nsd_id": "nsd_id_i",
            "subject_index": "subject_index_i",
            "repetition": "repetition_i",
        }
    )
    df_j = df_j[["subject", "nsd_id", "subject_index", "repetition"]]
    if shift:
        df_j["repetition"] = (df_j["repetition"] + shift) % 3
    df_j = df_j.rename(
        columns={
            "subject": "subject_j",
            "nsd_id": "nsd_id_j",
            "subject_index": "subject_index_j",
            "repetition": "repetition_j",
        }
    )
    df_merged = df_i.merge(
        df_j,
        left_on=["nsd_id_i", "repetition_i"],
        right_on=["nsd_id_j", "repetition_j"],
    )
    return df_merged


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
    rdms_subj_i: torch.Tensor,
    rdms_subj_j: torch.Tensor,
    diagonal: bool = True,
):
    """
    Computes RSA tensor with dimensions n_roi x n_roi.

    Args:
        subject_features: tensor of shape (n_roi, n_session, n_flat_rdm)
        model_features: tensor of shape (n_layer, n_session, n_flat_rdm)

    Returns:
        rsa_tensor: tensor of shape (n_roi, n_layer, n_session)
    """

    if diagonal:
        rsa_matrix = einsum(rdms_subj_i, rdms_subj_j)
    else:
        rsa_matrix = matprod(rdms_subj_i, rdms_subj_j)

    return rsa_matrix.cpu().numpy()


def unravel_tensor(
    tensor: np.ndarray,
    subject_i: int,
    subject_j: int,
    join_hemispheres: bool,
    shift: int,
    diagonal: bool = False,
) -> pd.DataFrame:
    # If diagonal it receives a vector [n_roi]
    # If not diagonal it receives a matrix [n_roi, n_roi]
    data = []
    if diagonal:
        for roi in range(tensor.shape[0]):
            data.append(
                {
                    "roi": roi + 1,  # ROI is indexed from 1
                    "similarity": float(tensor[roi]),
                }
            )
    else:
        for roi_x in range(tensor.shape[0]):
            for roi_y in range(tensor.shape[1]):
                data.append(
                    {
                        "roi_x": roi_x + 1,  # ROI is indexed from 1
                        "roi_y": roi_y + 1,  # ROI is indexed from 1
                        "similarity": float(tensor[roi_x, roi_y]),
                    }
                )

    df = pd.DataFrame(data)
    df.similarity = df.similarity.astype("float32")
    df["subject_i"] = subject_i
    df["subject_i"] = df["subject_i"].astype("uint8")
    df["subject_j"] = subject_j
    df["subject_j"] = df["subject_j"].astype("uint8")
    df["join_hemispheres"] = join_hemispheres
    df["join_hemispheres"] = df["join_hemispheres"].astype("bool")
    df["rep_shift"] = shift
    df["rep_shift"] = df["rep_shift"].astype("uint8").astype("category")

    if "roi" in df.columns:
        df["roi"] = df["roi"].astype("uint16")  # 1-360
    else:
        df["roi_x"] = df["roi_x"].astype("uint16")
        df["roi_y"] = df["roi_y"].astype("uint16")

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
    subject_features_i: torch.Tensor,  # [n_roi, n_flat_rdm_shape]
    subject_features_j: torch.Tensor,  # [n_roi, n_flat_rdm_shape]
    df_control: pd.DataFrame,
    nsd_ids,
    diagonal: bool = False,
    distance="euclidean",
) -> pd.DataFrame:

    rdm_control = control_pandas_to_rdm(df_control, nsd_ids=nsd_ids, distance=distance)

    r_AC = torch.einsum("ij,j->i", subject_features_i, rdm_control)
    r_BC = torch.einsum("ij,j->i", subject_features_j, rdm_control)

    if diagonal:
        r_AB = torch.einsum("ij,ij->i", subject_features_i, subject_features_j)
        partial_r = (r_AB - r_AC * r_BC) / torch.sqrt((1 - r_AC**2) * (1 - r_BC**2))
    else:  # Compute all pairwise rois correspondences
        r_AB = torch.mm(subject_features_i, subject_features_j.T)
        partial_r = (r_AB - r_AC[:, None] * r_BC[None, :]) / torch.sqrt(
            (1 - r_AC[:, None] ** 2) * (1 - r_BC[None, :] ** 2)
        )

    return partial_r


def compare_subject_subject(
    subject_i: int,
    subject_j: int,
    join_hemisphere: bool,
    control_files: list[Path],
    control_nsd_ids: list[int],
    shift: int = 1,
    diagonal: bool = False,
    control_distance: str = "euclidean",
):

    df_merge = get_common_indexes(
        subject_i=subject_i,
        subject_j=subject_j,
        shift=shift,
        restricted_nsd_ids=control_nsd_ids,
    )
    subject_indexes_i = df_merge.subject_index_i.values
    subject_features_i = prepare_subject_features(
        subject=subject_i,
        subject_indexes=subject_indexes_i,
        join_hemisphere=join_hemisphere,
    )

    subject_indexes_j = df_merge.subject_index_j.values
    subject_features_j = prepare_subject_features(
        subject=subject_j,
        subject_indexes=subject_indexes_j,
        join_hemisphere=join_hemisphere,
    )

    # Compute the uncontrolled RSA tensor
    rsa_tensor = compute_rsa_tensor(
        subject_features_i, subject_features_j, diagonal=diagonal
    )
    df_uncontrolled = unravel_tensor(
        tensor=rsa_tensor,
        subject_i=subject_i,
        subject_j=subject_j,
        join_hemispheres=join_hemisphere,
        shift=shift,
        diagonal=diagonal,
    )
    df_uncontrolled["control"] = "uncontrolled"
    df_uncontrolled["distance"] = "uncontrolled"
    dfs = [df_uncontrolled]

    # Control RDM
    for control_file in tqdm(control_files, desc="Controls", position=3, leave=False):
        df_control = pd.read_parquet(control_file)
        control_name = control_file.stem.replace("control_", "")
        rsa_tensor_controled = compute_controled_rsa(
            subject_features_i=subject_features_i,
            subject_features_j=subject_features_j,
            df_control=df_control,
            diagonal=diagonal,
            nsd_ids=df_merge.nsd_id_i.values,
            distance=control_distance,
        )
        df_controled = unravel_tensor(
            tensor=rsa_tensor_controled,
            subject_i=subject_i,
            subject_j=subject_j,
            join_hemispheres=join_hemisphere,
            shift=shift,
            diagonal=diagonal,
        )
        df_controled["control"] = control_name
        df_controled["distance"] = control_distance
        dfs.append(df_controled)


    dfs = pd.concat(dfs, ignore_index=True)
    dfs["control"] = dfs["control"].astype("category")
    dfs["distance"] = dfs["distance"].astype("category")
    return dfs


@alert(disable=TEST)
def main():
    args = parse_args()
    output_filename = Path(args.output_filename)
    join_hemisphere = args.join_hemispheres
    n_subjects = get_resource("stimulus").subject.nunique()
    if TEST:
        print("TESTING")
        n_subjects = 1  # TEST
    shift = args.shift
    diagonal = args.diagonal

    # Control files
    control_folder = Path("control_matrices")
    control_files = list(control_folder.glob("control_*.parquet"))
    # Assume all control files have the same nsd_ids
    restricted_nsd_ids = pd.read_parquet(control_files[0]).nsd_id.unique()

    for subject_i in trange(1, n_subjects + 1, position=0, desc="Subj-i", leave=False):
        if not TEST:
            send_alert(f"Processing subject {subject_i}")
        subject_results = []
        subject_filename = Path(f"{output_filename}_{subject_i}.parquet")
        if subject_filename.exists() and not TEST:
            continue
        for subject_j in trange(
            1, n_subjects + 1, position=1, desc="Subj-j", leave=False
        ):
            df_model_subject = compare_subject_subject(
                subject_i=subject_i,
                subject_j=subject_j,
                join_hemisphere=join_hemisphere,
                shift=shift,
                diagonal=diagonal,
                control_files=control_files,
                control_nsd_ids=restricted_nsd_ids,
            )
            subject_results.append(df_model_subject)
            gc.collect()
            torch.cuda.empty_cache()

        subject_results = pd.concat(subject_results)
        subject_results.to_parquet(subject_filename, index=False)
        del subject_results
        gc.collect()


if __name__ == "__main__":
    main()

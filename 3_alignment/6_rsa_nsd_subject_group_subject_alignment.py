"""
rsa_nsd_subject_group_subject_alignment.py

Compute inter-subject representational similarity (IS-RSA) between each NSD subject and 
the group average (excluding the subject) across cortical ROIs. It compares representational 
dissimilarity matrices (RDMs) from fMRI beta activations across shared stimuli.

Features:
- Computes ROI-wise RSA (Pearson or Spearman) between a subject and the group.
- Supports comparison using diagonal (per-ROI) or full cross-ROI similarity matrices.
- Optionally performs permutation testing to generate null distributions.
- Handles joined (180) or separated (360) hemisphere ROI configurations.

Usage:
    python rsa_nsd_subject_group_subject_alignment.py [options]

Main Arguments:
    --output_filename         Output .parquet file to save results. If not provided, auto-generated.
    --join_hemispheres        If set, merges left/right hemisphere ROIs into 180 combined ROIs.
    --shift                   Repetition alignment shift for group comparisons (default: 1).
    --diagonal                If set, compares only matching ROIs across subject and group.
    --n_permutations          Number of permutations for null distribution (optional).
    --permutations_folder     Folder to store/reuse generated permutation files.
    --spearman                Use Spearman instead of Pearson for RSA.
    --signal-to-noise         Compute SNR from mean RDM (not used directly in this script).

Outputs:
- A `.parquet` file with ROI-wise or cross-ROI similarity scores between each subject and the group.
- If `--n_permutations` is set, saves a corresponding `.npy` file with null distributions.

Dependencies:
- Requires preprocessed NSD betas and metadata indexed by the `convergence` library.
- Uses GPU (`cuda`) for efficient tensor computation.

Example:
    python rsa_nsd_subject_group_subject_alignment.py --join_hemispheres --shift 1 --diagonal --n_permutations 1000

"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import trange, tqdm

import gc
import pandas as pd

from dmf.alerts import alert, send_alert, send_message
from convergence.nsd import get_subject_roi, get_resource
from convergence.permutations import generate_batched_permutations

DEVICE = "cuda"
PERMUTATION_BATCH_SIZE = 5
TEST = False

def rank_tensor(x):
    """
    Returns a tensor of the same shape as x containing the rank of each element.
    Ranks are 0-based by default.
    Ties are broken arbitrarily.
    """
    x_flat = x.view(-1)
    # sort once
    _, sorted_idx = x_flat.sort()
    # allocate rank array
    ranks = torch.empty_like(sorted_idx)
    # place 0,1,2,... at the sorted positions
    ranks[sorted_idx] = torch.arange(x_flat.size(0), device=x.device)
    ranks = ranks.view_as(x)
    ranks = ranks / ranks.size(0)
    return ranks.to(x.dtype)


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
    parser.add_argument(
        "--spearman",
        action="store_true",
        help="Use Spearman correlation instead of Pearson correlation.",
    )
    parser.add_argument(
        "--signal-to-noise",
        action="store_true",
        help="Compute signal-to-noise using mean RDM instead of all comparison",
    )

    # Parse the arguments
    args = parser.parse_args()
    if args.output_filename is None:
        spearman_suffix = "_spearman" if args.spearman else ""
        joined_suffix = "joined" if args.join_hemispheres else "separated"
        diagonal_suffix = "_diagonal" if args.diagonal else ""
        signal_to_noise_suffix = "_signal_to_noise" if args.signal_to_noise else ""

        args.output_filename = (
            f"rsa_subject_group_alignment_{joined_suffix}_{args.shift}"
        )
        args.output_filename += (
            f"{diagonal_suffix}{spearman_suffix}{signal_to_noise_suffix}"
        )

    if not args.output_filename.endswith(".parquet"):
        args.output_filename += ".parquet"

    return args


def get_common_indexes(subject_i, subject_j, shift=0):

    df = get_resource("stimulus")
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


def get_shared_stimulus_subset(subject_i, shift=0):
    df_stimuli = get_resource("stimulus")
    # Get n_subjects
    subjects = list(df_stimuli.subject.unique())

    df_shared = df_stimuli.query("subject == @subject_i and shared and exists")
    df_shared = df_shared[["nsd_id", "repetition", "subject_index"]]
    subject_pairs = []
    for subject_j in subjects:
        if subject_j == subject_i:
            continue

        df_pair = get_common_indexes(subject_i, subject_j, shift=shift)
        df_pair = df_pair[["subject_index_i", "subject_index_j"]]
        df_pair = df_pair.rename(
            columns={
                "subject_index_i": "subject_index",
                "subject_index_j": f"subject_index_{subject_j}",
            }
        )
        subject_pairs.append(df_pair)
        df_shared = df_shared.merge(df_pair, on="subject_index", how="inner")

    return df_shared


def create_flat_normalize_rdm(
    rdm: torch.Tensor, triu_indices: torch.Tensor = None, spearman=False
) -> torch.Tensor:
    if triu_indices is None:
        triu_indices = torch.triu_indices(rdm.size(0), rdm.size(0), offset=1)
    rdm_flat = rdm[triu_indices[0], triu_indices[1]]

    if spearman:
        rdm_flat = rank_tensor(rdm_flat)

    rdm_flat = rdm_flat - rdm_flat.mean()
    rdm_flat /= rdm_flat.norm()
    return rdm_flat


def compute_flat_rdm(betas_subset: np.ndarray, q=0.003, spearman=False) -> torch.Tensor:
    a, b = np.quantile(betas_subset, [q, 1 - q])
    betas_subset = np.clip(betas_subset, a, b)

    betas_subset = (betas_subset - a) / (b - a)
    betas_subset = torch.tensor(betas_subset, device=DEVICE, dtype=torch.float32)

    # Compute RDM
    betas_subset = betas_subset - betas_subset.mean(dim=1, keepdim=True)
    betas_subset = torch.nn.functional.normalize(betas_subset, dim=1)
    rdm = 1 - torch.mm(betas_subset, betas_subset.t())
    return create_flat_normalize_rdm(rdm, spearman=spearman)


def prepare_subject_features(
    subject, subject_indexes, join_hemisphere: bool, spearman: bool = False
):

    total_rois = 180 if join_hemisphere else 360
    if TEST:
        total_rois = 4

    roi_session_rdms = []

    for roi in trange(
        1, total_rois + 1, desc="Preparing ROIs", position=2, leave=False
    ):
        roi_betas = get_subject_roi(
            subject, roi if not join_hemisphere else [roi, roi + 180]
        )

        flat_rdm = compute_flat_rdm(roi_betas[subject_indexes], spearman=spearman)
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
    # df["subject_j"] = df["subject_j"].astype("uint8")
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


@torch.jit.script
def optimized_permutation(
    perm: torch.Tensor,
    features_i: torch.Tensor,
    features_j: torch.Tensor,
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

    result = torch.einsum("bf,bf->b", features_i, features_j[:, order])

    return result


def get_n_from_vector_size(vector_size: int) -> int:
    return int((1 + (1 + 8 * vector_size) ** 0.5) / 2)


def restrict_permutation(perm, n_common):
    return perm[perm < n_common]


def compute_permutations(
    features_i: torch.Tensor,
    features_j: torch.Tensor,
    permutations: torch.Tensor,
):
    n_permutations, maximal_n = permutations.shape
    n_roi, N = features_i.shape
    n = get_n_from_vector_size(N)
    device = features_i.device

    rsa_permutations = torch.zeros(n_permutations, n_roi, device=device)

    # Allocate these once outside the loop
    triu_i, triu_j = torch.triu_indices(n, n, offset=1, device=device)
    inv_perm = torch.empty(n, dtype=torch.long, device=device)
    order = torch.empty(N, dtype=torch.long, device=device)
    arange_n = torch.arange(n, device=device)
    arange_N = torch.arange(N, device=device)

    for i in trange(n_permutations, desc="Permutations", position=3, leave=False):

        permutations_i = permutations[i]
        if maximal_n > n:
            permutations_i = restrict_permutation(permutations_i, n)

        rsa_permutations[i] = optimized_permutation(
            perm=permutations_i.to(device),
            features_i=features_i,
            features_j=features_j,
            triu_i=triu_i,
            triu_j=triu_j,
            inv_perm=inv_perm,
            order=order,
            arange_n=arange_n,
            arange_N=arange_N,
            n=n,
            N=N,
        )

    return rsa_permutations.cpu()


def prepare_group_features(
        df_merge,
        subject_i,
        join_hemisphere=True,
        spearman=False
):
    df_stimuli = get_resource("stimulus")
    subjects = df_stimuli.subject.unique()
    total_group = len(subjects) - 1  # Exclude subject_i
    group_features = None

    for subject_j in tqdm(subjects, desc="Group features", position=1, leave=False):
        if subject_j == subject_i:
            continue 

        subject_indexes_j = df_merge[f"subject_index_{subject_j}"].values
        subject_features_j = prepare_subject_features(
            subject=subject_j,
            subject_indexes=subject_indexes_j,
            join_hemisphere=join_hemisphere,
            spearman=spearman,
        )
        if group_features is None:
            group_features = subject_features_j
        else:
            group_features += subject_features_j
    # Average the features across all subjects in the group
    group_features /= total_group
    return group_features

def compare_subject_group(
    subject_i: int,
    join_hemisphere: bool,
    shift: int = 1,
    diagonal: bool = False,
    permutations: torch.Tensor = None,
    permutations_folder: Path = None,
    spearman: bool = False,
):
    df_merge = get_shared_stimulus_subset(subject_i=subject_i, shift=shift)

    subject_indexes_i = df_merge.subject_index.values
    subject_features_i = prepare_subject_features(
        subject=subject_i,
        subject_indexes=subject_indexes_i,
        join_hemisphere=join_hemisphere,
        spearman=spearman,
    )


    group_features = prepare_group_features(
        df_merge,
        subject_i=subject_i,
        join_hemisphere=join_hemisphere,
        spearman=spearman,
    )

    if permutations is not None:
        n_permutations = permutations.size(0)
        permutation_results = compute_permutations(
            features_i=subject_features_i,
            features_j=group_features,
            permutations=permutations,
        )
        permutation_file = (
            permutations_folder
            / f"subject_subject_permutations_{subject_i}_group_{shift}_{n_permutations}.npy"
        )
        # Save as npy file
        np.save(permutation_file, permutation_results.numpy())
        del permutation_results

    rsa_tensor = compute_rsa_tensor(
        subject_features_i, group_features, diagonal=diagonal
    )

    return unravel_tensor(
        tensor=rsa_tensor,
        subject_i=subject_i,
        subject_j=f"group_without_{subject_i}",
        join_hemispheres=join_hemisphere,
        shift=shift,
        diagonal=diagonal,
    )


@alert
def main():
    args = parse_args()
    output_filename = Path(args.output_filename)
    join_hemisphere = args.join_hemispheres
    n_subjects = 8
    max_shared_trials = 3000
    shift = args.shift
    diagonal = args.diagonal
    perms = args.n_permutations
    permutations_folder = Path(args.permutations_folder)
    spearman = args.spearman

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
    subject_results = []
    for subject_i in trange(1, n_subjects + 1, position=0, desc="Subj-i", leave=False):

        send_alert(f"Processing subject {subject_i}")
        
        df_model_subject = compare_subject_group(
            subject_i=subject_i,
            join_hemisphere=join_hemisphere,
            shift=shift,
            diagonal=diagonal,
            permutations=permutations,
            permutations_folder=permutations_folder,
            spearman=spearman,
        )
        subject_results.append(df_model_subject)
        gc.collect()
        torch.cuda.empty_cache()

    subject_results = pd.concat(subject_results)
    subject_results.to_parquet(output_filename, index=False)
    send_message("Finished group IS-RSA measures for ceil", attachment=output_filename)


if __name__ == "__main__":
    main()

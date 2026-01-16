import argparse
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import torch
from tqdm import trange, tqdm
from dmf.alerts import alert, send_alert

from convergence.nsd import get_subject_roi, get_resource

DEVICE = "cuda"


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
        "--spearman",
        action="store_true",
        help="Use Spearman correlation instead of Pearson correlation.",
    )
    parser.add_argument(
        "--n_subsets",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="rsa_reliability_isrsa.parquet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--stepwise",
        action="store_true",
        help="Generate subsets with stepwise sizes.",
    )
    parser.add_argument(
        "--min_subset_size",
        type=int,
        default=50,
        help="Minimum subset size for stepwise generation.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=50,
        help="Step size for subset size in stepwise generation.",
    )
    return parser.parse_args()


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


def create_rdm(betas_subset: np.ndarray, q=0.003, spearman=False) -> torch.Tensor:
    a, b = np.quantile(betas_subset, [q, 1 - q])
    betas_subset = np.clip(betas_subset, a, b)

    betas_subset = (betas_subset - a) / (b - a)
    betas_subset = torch.tensor(betas_subset, device=DEVICE, dtype=torch.float32)

    # Compute RDM
    betas_subset = betas_subset - betas_subset.mean(dim=1, keepdim=True)
    betas_subset = torch.nn.functional.normalize(betas_subset, dim=1)
    rdm = 1 - torch.mm(betas_subset, betas_subset.t())
    return rdm


def generate_subsets(n_subsets: int = 100, seed: int = 42, subset_size: int = 500) -> pd.DataFrame:
    """Generate random subsets of stimuli."""
    np.random.seed(seed)
    stimuli = get_resource("stimulus").query("shared and exists").nsd_id.unique()

    # Generate n_subset choices with replacement of subset_size stimuli
    subsets = []
    for subset_i in range(n_subsets):
        subset_stimuli = np.random.choice(stimuli, size=subset_size, replace=False)
        subsets.append({"subset": subset_i, "nsd_id": subset_stimuli})

    return pd.DataFrame(subsets)

def generate_subsets_stepwise(n_subsets: int = 100, seed: int = 42, step_size: int = 50, min_subset_size: int = 50, max_subset_size: int = 700) -> pd.DataFrame:
    """Generate random subsets of stimuli with stepwise sizes."""
    np.random.seed(seed)
    stimuli = get_resource("stimulus").query("shared and exists").nsd_id.unique()

    # Generate subsets with varying sizes
    subsets = []
    for subset_size in range(min_subset_size, max_subset_size + 1, step_size):
        for subset_i in range(n_subsets):
            subset_stimuli = np.random.choice(stimuli, size=subset_size, replace=False)
            subsets.append({"subset": f"{subset_i}_size_{subset_size}", "nsd_id": subset_stimuli})

    return pd.DataFrame(subsets)


def compare_subject_subject(
    subject_i: int,
    subject_j: int,
    df_subsets: pd.DataFrame,
    join_hemisphere: bool = True,
    shift: int = 1,
    spearman: bool = False,
    q=0.003,
):
    df_merge = get_common_indexes(subject_i=subject_i, subject_j=subject_j, shift=shift)
    subject_i_indexes = df_merge["subject_index_i"].values
    subject_j_indexes = df_merge["subject_index_j"].values
    nsd_ids = df_merge["nsd_id_i"].values
    pair_results = []


    total_rois = 180 if join_hemisphere else 360
    for roi in trange(1, total_rois + 1, desc="ROIs", position=2, leave=False):
        roi_betas_i = get_subject_roi(subject_i, roi if not join_hemisphere else [roi, roi + 180])
        roi_betas_i = roi_betas_i[subject_i_indexes]
        roi_betas_j = get_subject_roi(subject_j, roi if not join_hemisphere else [roi, roi + 180])
        roi_betas_j = roi_betas_j[subject_j_indexes]

        rdm_i = create_rdm(roi_betas_i, spearman=spearman, q=q)
        rdm_j = create_rdm(roi_betas_j, spearman=spearman, q=q)
 

        for _, row in (pbar:=tqdm(df_subsets.iterrows(), total=len(df_subsets), desc="Subsets", position=3, leave=False)):

            subset_nsd_ids = row["nsd_id"]
            mask = np.isin(nsd_ids, subset_nsd_ids)
            mask = torch.tensor(mask, device=DEVICE)


            rdm_i_subset = rdm_i[mask][:, mask]
            rdm_j_subset = rdm_j[mask][:, mask]


            rdm_i_flat = create_flat_normalize_rdm(
                rdm_i_subset, spearman=spearman, 
            )
            rdm_j_flat = create_flat_normalize_rdm(
                rdm_j_subset, spearman=spearman, 
            )

            similarity = torch.dot(rdm_i_flat, rdm_j_flat).item()
            subset_size = mask.sum().item()
            pair_results.append(
                {
                    "roi": roi,
                    "subset": row["subset"],
                    "subset_size": subset_size,
                    "similarity": similarity,
                }
            )
            pbar.set_postfix({"last_similarity": f"{similarity:.3f}",
                              "subset_size": f"{subset_size}"})

    df_pair_results = pd.DataFrame(pair_results)
    df_pair_results["subject_i"] = subject_i
    df_pair_results["subject_j"] = subject_j
    return df_pair_results


@alert
def main():
    args = parse_args()
    output_filename = Path(args.output_file)
    join_hemisphere = args.join_hemispheres
    shift = args.shift
    spearman = args.spearman
    n_subsets = args.n_subsets
    seed = args.seed
    subset_size = args.subset_size
    n_subjects = 8

    if args.stepwise:
        df_subsets = generate_subsets_stepwise(n_subsets=n_subsets, seed=seed, step_size=args.step_size, min_subset_size=args.min_subset_size, 
                                                max_subset_size=args.subset_size)
        df_subsets.to_parquet(output_filename.parent / f"subsets-stepwise-{seed}.parquet", index=False)
    else:

        df_subsets = generate_subsets(n_subsets=n_subsets, seed=seed, subset_size=subset_size)
        df_subsets.to_parquet(output_filename.parent / f"subsets-{seed}.parquet", index=False)

    for subject_i in trange(1, n_subjects + 1, position=0, desc="Subj-i", leave=False):
        send_alert(f"Processing subject {subject_i}")
        subject_results = []
        subject_filename = Path(output_filename.stem + f".{seed}-subject_{subject_i}.parquet")
        if subject_filename.exists():
            continue
        for subject_j in trange(1, n_subjects + 1, position=1, desc="Subj-j", leave=False):
            df_model_subject = compare_subject_subject(
                subject_i=subject_i,
                subject_j=subject_j,
                join_hemisphere=join_hemisphere,
                shift=shift,
                spearman=spearman,
                df_subsets=df_subsets,
            )
            if df_model_subject is None:  # Testing
                break
            subject_results.append(df_model_subject)
            gc.collect()
            torch.cuda.empty_cache()

        subject_results = pd.concat(subject_results)
        subject_results.to_parquet(subject_filename, index=False)
        del subject_results
        gc.collect()


if __name__ == "__main__":
    main()

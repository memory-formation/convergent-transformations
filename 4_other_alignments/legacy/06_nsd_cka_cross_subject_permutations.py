"""
Legacy script to compute cross-subject CKA with permutation testing using full-brain ROI loops.

For each pair of NSD subjects and ROIs, the script:
- Extracts beta activations for shared images (with optional repetition shift).
- Computes unbiased CKA using HSIC.
- Builds a null distribution by permuting the stimulus order.
- Saves a DataFrame with observed and permuted CKA values.

Optionally merges hemispheres (180 ROIs) or analyzes them separately (360 ROIs).
"""

import argparse
from pathlib import Path

from tqdm import trange
import pandas as pd
import numpy as np
import torch

from convergence.nsd import get_subject_roi, get_resource
from convergence.metrics.utils import hsic_unbiased
from dmf.alerts import alert, send_alert


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

def center_kernel_matrix_efficient(K):
    """
    Centers the NxN kernel matrix K in a single pass without 
    explicitly forming H = I - 1/n 11^T.
    """
    row_mean = K.mean(dim=1, keepdim=True)       # shape (N, 1)
    col_mean = K.mean(dim=0, keepdim=True)       # shape (1, N)
    grand_mean = K.mean()                        # scalar
    
    # apply: K - row_mean - col_mean + grand_mean
    K_centered = K - row_mean - col_mean + grand_mean
    return K_centered


def get_subject_betas(subject, roi, subject_index, q=0.003):
    betas = get_subject_roi(subject=subject, roi=roi)
    betas = betas[subject_index]
    a, b = np.quantile(betas.ravel(), [q, 1 - q])
    betas = np.clip(betas, a, b)
    betas = (betas - a) / (b - a)
    betas = torch.tensor(betas, device="cuda", dtype=torch.float32)
    return betas


def compare_subjects(
    subject_i, subject_j, shift=0, n_repetitions=1000, join_hemispheres=True
):
    pair_results = []
    total_rois = 180 if join_hemispheres else 360
    df_common_indexes = get_common_indexes(
        subject_i=subject_i, subject_j=subject_j, shift=shift
    )
    indexes_i = df_common_indexes.subject_index_i.tolist()
    indexes_j = df_common_indexes.subject_index_j.tolist()
    for roi in trange(1, total_rois + 1, desc="ROI", leave=False, position=2):
        roi_key = [roi, roi + 180] if join_hemispheres else roi
        betas_i = get_subject_betas(
            subject=subject_i, roi=roi_key, subject_index=indexes_i
        )
        betas_j = get_subject_betas(
            subject=subject_j, roi=roi_key, subject_index=indexes_j
        )
        assert betas_i.shape[0] == betas_j.shape[0], "Different number stimuli"
        n_stimuli = betas_i.shape[0]

        K = torch.mm(betas_i, betas_i.T)
        K = center_kernel_matrix_efficient(K)
        L = torch.mm(betas_j, betas_j.T)
        L = center_kernel_matrix_efficient(L)

        hsic_kk = hsic_unbiased(K, K)
        hsic_ll = hsic_unbiased(L, L)
        normalization_term = (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)

        for repetition in trange(
            0, n_repetitions + 1, desc="Repetition", leave=False, position=3
        ):

            if repetition == 0:
                hsic_kl = hsic_unbiased(K, L)
            else:
                idx = torch.randperm(n_stimuli)
                hsic_kl = hsic_unbiased(K, L[idx][:, idx])

            cka_value = hsic_kl / normalization_term
            pair_results.append(
                {
                    "subject_i": subject_i,
                    "subject_j": subject_j,
                    "roi": roi,
                    "repetition": repetition,
                    "permuted": repetition > 0,
                    "cka": float(cka_value.item()),
                }
            )

    return pair_results


@alert(input=True, output=True)
def compute_permutations(
    n_repetitions=1000,
    shift=0,
    output="nsd_cka_cross_subject_permutations.parquet",
    join_hemispheres=True,
):
    results = []

    n_participants = 8
    for i in trange(1, n_participants + 1, desc="Subject i", leave=False, position=0):
        send_alert(f"Processing subject {i}")
        for j in trange(
            1, n_participants + 1, desc="Subject j", leave=False, position=1
        ):
            if i >= j:
                continue
            pair_result = compare_subjects(
                subject_i=i,
                subject_j=j,
                shift=shift,
                n_repetitions=n_repetitions,
                join_hemispheres=join_hemispheres,
            )
            results.extend(pair_result)

    results = pd.DataFrame(results)
    results.to_parquet(output)
    return Path(output)


def main():
    argparser = argparse.ArgumentParser()
    # N-repetitions
    argparser.add_argument("--n_reps", type=int, default=1000)
    # Shift
    argparser.add_argument("--shift", type=int, default=0)
    argparser.add_argument(
        "--output", type=str, default="nsd_cka_cross_subject_permutations.parquet"
    )
    argparser.add_argument("--separated_hemispheres", action="store_true")
    args = argparser.parse_args()
    compute_permutations(
        n_repetitions=args.n_reps,
        shift=args.shift,
        output=args.output,
        join_hemispheres=not args.separated_hemispheres,
    )


if __name__ == "__main__":
    main()

"""
Legacy script to compute cross-subject CKA between ROI pairs using permutation testing.

For each subject pair and ROI edge, the script:
- Extracts aligned beta responses for common stimuli.
- Computes unbiased CKA using HSIC.
- Estimates a null distribution via random permutations of stimuli.

Used to quantify representational similarity and test statistical significance
of cross-subject functional connectivity patterns.
"""

import argparse
from pathlib import Path
import os
from tqdm import trange, tqdm
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
    row_mean = K.mean(dim=1, keepdim=True)  # shape (N, 1)
    col_mean = K.mean(dim=0, keepdim=True)  # shape (1, N)
    grand_mean = K.mean()  # scalar

    # apply: K - row_mean - col_mean + grand_mean
    K_centered = K - row_mean - col_mean + grand_mean
    return K_centered


def get_subject_betas(subject, roi, subject_index, q=0.003):
    betas = get_subject_roi(subject=subject, roi=roi)
    betas = betas[subject_index]
    a, b = np.quantile(betas.ravel(), [q, 1 - q])
    betas = np.clip(betas, a, b)
    betas = (betas - a) / (b - a)
    betas = torch.tensor(betas, device="cuda", dtype=torch.float32) # Float64 for high precision
    return betas


def compare_subjects(
    edge_list, subject_i, subject_j, shift=0, n_repetitions=1000, join_hemispheres=True
):
    pair_results = []
    df_common_indexes = get_common_indexes(
        subject_i=subject_i, subject_j=subject_j, shift=shift
    )
    indexes_i = df_common_indexes.subject_index_i.tolist()
    indexes_j = df_common_indexes.subject_index_j.tolist()


    for (roi_x, roi_y) in (pbar := tqdm(edge_list, leave=False, position=2)):
        pbar.set_description(f"ROIs: {roi_x} - {roi_y}")
        if join_hemispheres:
            roi_x_key = [roi_x, roi_x + 180]
            roi_y_key = [roi_y, roi_y + 180]
        else:
            roi_x_key = roi_x
            roi_y_key = roi_y


        betas_i = get_subject_betas(
            subject=subject_i, roi=roi_x_key, subject_index=indexes_i
        )
        betas_j = get_subject_betas(
            subject=subject_j, roi=roi_y_key, subject_index=indexes_j
        )
        assert betas_i.shape[0] == betas_j.shape[0], "Different number stimuli"
        n_stimuli = betas_i.shape[0]

        K = torch.mm(betas_i, betas_i.T)
        #K = center_kernel_matrix_efficient(K)# Not needed. Do it in hsic_unbiased
        L = torch.mm(betas_j, betas_j.T)
        #L = center_kernel_matrix_efficient(L)

        hsic_kk = hsic_unbiased(K, K)
        hsic_ll = hsic_unbiased(L, L)
        normalization_term = torch.sqrt(hsic_kk * hsic_ll) + 1e-6

        for repetition in trange(
            0, n_repetitions + 1, desc="Permutation", leave=False, position=3
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
                    "roi_x": roi_x,
                    "roi_y": roi_y,
                    "repetition": repetition,
                    "permuted": repetition > 0,
                    "shift": shift,
                    "n_stimuli": n_stimuli,
                    "cka": float(cka_value.item()),
                }
            )

    return pair_results


@alert(input=True, output=True)
def compute_permutations(
    edge_list,
    n_repetitions=1000,
    shift=0,
    output="nsd_cka_cross_subject_permutations_edges.parquet",
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
                edge_list=edge_list,
                subject_i=i,
                subject_j=j,
                shift=shift,
                n_repetitions=n_repetitions,
                join_hemispheres=join_hemispheres,
            )
            results.extend(pair_result)

    results = results_to_pandas(results)
    results.to_parquet(output)
    return Path(output)

def results_to_pandas(results):
    df = pd.DataFrame(results)
    # from 1 to 8
    df.subject_i = df.subject_i.astype("int8")
    df.subject_j = df.subject_j.astype("int8")
    # from 1 to 180
    df.roi_x = df.roi_x.astype("int16")
    df.roi_y = df.roi_y.astype("int16")
    # from 0 to n_repetitions
    df.repetition = df.repetition.astype("int32")
    # Shift constant number 0, 1,2
    df["shift"] = df["shift"].astype("int8")
    # Number of stimuli costant number less than 10K
    df.n_stimuli = df.n_stimuli.astype("int16").astype("category")
    # CKA value
    df.cka = df.cka.astype("float32")
    # Permuted boolean
    df.permuted = df.permuted.astype("bool")

    return df


def read_edges(edge_list):
    edges = pd.read_parquet(edge_list)
    edges = (
        edges[["roi_x", "roi_y"]].drop_duplicates().sort_values(["roi_x", "roi_y"])
    )  # .to_dict("records")
    edges = edges.query("roi_x < roi_y").to_numpy().astype(int).tolist()

    return edges


def main():
    argparser = argparse.ArgumentParser()
    # N-repetitions
    argparser.add_argument("--n_reps", type=int, default=5000)
    # Shift
    argparser.add_argument("--shift", type=int, default=0)
    argparser.add_argument(
        "--output", type=str, default="nsd_cka_cross_subject_permutations_edges.parquet"
    )
    argparser.add_argument("--separated_hemispheres", action="store_true")
    argparser.add_argument("--edge_list", type=str, default=None)
    args = argparser.parse_args()

    if args.edge_list is None:
        results = Path(os.getenv("CONVERGENCE_RESULTS"))
        edge_list = results / "graph/graph_top_edges_20_rois.parquet"
    else:
        edge_list = Path(args.edge_list)

    edges = read_edges(edge_list)

    compute_permutations(
        edge_list=edges,
        n_repetitions=args.n_reps,
        shift=args.shift,
        output=args.output,
        join_hemispheres=not args.separated_hemispheres,
    )


if __name__ == "__main__":
    main()

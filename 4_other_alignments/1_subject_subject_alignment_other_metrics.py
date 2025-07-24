"""
subject_subject_alignment_other_metrics.py

This script computes pairwise representational alignment between brain subjects using
multiple similarity metrics across ROIs. It generalizes inter-subject RSA to also support:

- RSA (Pearson correlation-based RDM comparison)
- CKA (Centered Kernel Alignment)
- Unbiased CKA
- Mutual k-Nearest Neighbors (mutual_knn)

The alignment is computed between all pairs of subjects (or optionally only diagonals),
across all ROIs (or joined hemispheres), and across repetitions (with optional shift).

### Usage

Run from command line:
    python subject_subject_alignment_other_metrics.py --metric rsa unbiased_cka --shift 1 --join_hemispheres

### Arguments
- `--metric`:
    List of metrics to compute. Choices: `rsa`, `cka`, `unbiased_cka`, `mutual_knn`.
- `--shift`:
    Integer (0–2). Shift repetition index of one subject (cyclically) to test temporal misalignment.
- `--output_filename`:
    Path to output `.parquet` file. If not provided, generated from args.
- `--join_hemispheres`:
    Merge left and right hemisphere ROIs into a single vector. Default: False.
- `--diagonal`:
    If True, computes only diagonal ROI comparisons (same ROI in both subjects).

### Output

A `.parquet` file containing a long-form table of inter-subject alignment values with the following fields:
- `subject_i`, `subject_j`: Subject IDs
- `roi_x`, `roi_y`: ROI indices
- `metric`: Name of similarity metric
- `score`: Computed similarity value
- `shift`: Repetition shift used
- `join_hemispheres`: Whether hemispheres were joined

### Notes
- Requires ROI-wise betas via `convergence.nsd.get_subject_roi`.
- Uses quantile clipping and normalization before similarity computation.
- Supports caching to minimize memory usage and runtime.

"""


import argparse
from pathlib import Path
from dmf.alerts import alert, send_alert
import pandas as pd
from tqdm import trange, tqdm
from convergence.nsd import get_index
from convergence.nsd import get_subject_roi
from convergence.metrics import measure
import numpy as np
import torch
import gc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cross-subject similarities pairwise distances"
    )

    # Argument for metric
    parser.add_argument(
        "--metric",
        nargs="+",  # Allows multiple choices
        choices=["rsa", "unbiased_cka", "mutual_knn", "cka"],
        default=["unbiased_cka"],  # Default is all metrics
        help="List of metrics to compute. Choices: rsa, unbiased_cka, mutual_knn, cka. Default is CKA.",
    )

    # Argument for shift
    parser.add_argument(
        "--shift",
        type=int,
        choices=range(0, 3),  # Limit shift to integers 0, 1, 2
        default=0,  # Default value
        help="Shift value (integer between 0 and 3). Default is 1.",
    )

    # Argument for output filename
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,  # Default will be computed based on shift
        help="Output filename for the results. Default is based on the shift value.",
    )

    parser.add_argument(
        "--join_hemispheres",
        action="store_true",
        help="Whether to join the hemispheres of the brain. Default is False.",
    )
    parser.add_argument(
        "--diagonal",
        action="store_true",
        help="Whether to compute the diagonal of the matrix. Default is False.",
    )

    # Parse the arguments
    args = parser.parse_args()
    if args.join_hemispheres:
        hm = "joined"
    else:
        hm = "separated"
    metric = "_".join(args.metric)
    # Set default output filename if not provided
    if args.output_filename is None:
        args.output_filename = (
            f"cross_subject_pairwise_similarities_{args.shift}_{hm}_{metric}.parquet"
        )

    # If not end in .parquet, add it
    if not args.output_filename.endswith(".parquet"):
        args.output_filename += ".parquet"

    return args


def get_common_indexes(subject_i, subject_j, shift=0):

    df = get_index("stimulus")
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


def get_betas(
    subject: int,
    roi: int,
    subject_index: list[int],
    join_hemispheres: bool,
    q: float = 0.003,
) -> torch.tensor:

    if join_hemispheres:
        roi = [roi, roi + 180]

    betas = get_subject_roi(subject=subject, roi=roi)
    betas = betas[subject_index].astype("float32")
    q0, q1 = np.quantile(betas, [q, 1 - q])
    betas = np.clip(betas, q0, q1)
    betas = (betas - q0) / (q1 - q0)

    betas = torch.tensor(betas, device="cuda", dtype=torch.float32)
    return betas


def compare_betas(
    betas_i: torch.tensor, betas_j: torch.tensor, metrics: list[str], **info
) -> dict:
    results = []
    for metric in metrics:
        r = measure(metric_name=metric, feats_A=betas_i, feats_B=betas_j)
        results.append(dict(score=r, metric=metric, **info))

    return results


@alert(input=["subject_i", "subject_j", "metrics", "diagonal"])
def compute_pair_similarity(
    subject_i: int,
    subject_j: int,
    metrics: list[str],
    join_hemispheres: bool,
    shift: int,
    diagonal: bool = False,
):
    results = []
    # Compute similarity between subject_i and subject_j using the metric
    df_merged = get_common_indexes(subject_i, subject_j, shift=shift)
    subject_index_i = df_merged["subject_index_i"].tolist()
    subject_index_j = df_merged["subject_index_j"].tolist()

    total_rois = 181 if join_hemispheres else 361

    betas_j_cache = {}

    for roi_x in trange(1, total_rois, desc="ROI X", leave=False, position=2):
        betas_i = get_betas(
            subject=subject_i,
            roi=roi_x,
            subject_index=subject_index_i,
            join_hemispheres=join_hemispheres,
        )
        for roi_y in trange(1, total_rois, desc="ROI Y", leave=False, position=3):
            if diagonal and roi_x != roi_y:
                continue
            if roi_y in betas_j_cache:
                betas_j = betas_j_cache[roi_y]
            else:
                betas_j = get_betas(
                    subject=subject_j,
                    roi=roi_y,
                    subject_index=subject_index_j,
                    join_hemispheres=join_hemispheres,
                )
                if not diagonal: # Cache only if computing all ROIs vs all ROIs
                    betas_j_cache[roi_y] = betas_j
            info = dict(
                subject_i=subject_i,
                subject_j=subject_j,
                roi_x=roi_x,
                roi_y=roi_y,
                shift=shift,
                join_hemispheres=join_hemispheres,
            )
            r = compare_betas(betas_i=betas_i, betas_j=betas_j, metrics=metrics, **info)
            results.extend(r)
    del betas_i, betas_j, betas_j_cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return results


@alert(input=["shift", "join_hemispheres", "output_filename", "diagonal"])
def compute_cross_subject_pairwise_similarities(
    metrics: list[str], shift: int, join_hemispheres: bool, output_filename: Path, diagonal: bool = False
) -> Path:
    send_alert(
        f"Computing cross-subject pairwise similarities with {metrics} metric, "
        f"shift {shift}, and join_hemispheres {join_hemispheres}"
    )
    try:
        results = []
        for i in trange(1, 9, desc="Subject i", leave=True, position=0):
            for j in trange(1, 9, desc="Subject j", leave=False, position=1):
                r = compute_pair_similarity(
                    subject_i=i,
                    subject_j=j,
                    metrics=metrics,
                    join_hemispheres=join_hemispheres,
                    shift=shift,
                    diagonal=diagonal,
                )
                results.extend(r)

    except (Exception, KeyboardInterrupt) as e:
        df = pd.DataFrame(results)
        df.to_parquet("cross_subject_pairwise_similarities_temp.parquet")
        raise e
    if results:
        df = pd.DataFrame(results)
        df.to_parquet(output_filename)
    return output_filename


def main():
    args = parse_args()
    metrics = args.metric
    shift = args.shift
    join_hemispheres = args.join_hemispheres
    output_filename = args.output_filename
    diagonal = args.diagonal

    compute_cross_subject_pairwise_similarities(
        metrics=metrics,
        shift=shift,
        join_hemispheres=join_hemispheres,
        output_filename=output_filename,
        diagonal=diagonal,
    )


if __name__ == "__main__":
    main()

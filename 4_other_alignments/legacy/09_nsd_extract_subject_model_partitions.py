"""
Computes pairwise inter-subject alignment across all ROIs in the NSD dataset,
separately for different stimulus content partitions (e.g., motion, person, animal).

Main features:
- Supports multiple similarity metrics (RSA, unbiased CKA).
- Allows diagonal alignment (ROI_i == ROI_j) or full cross-ROI matrices.
- Stimulus filtering based on COCO categories and motion annotations.
- Output is a single parquet file with alignment scores per subject pair, ROI pair, and partition.

Data dependencies:
- Subject fMRI betas from NSD.
- COCO-style object annotations (`coco-objects`).
- Motion/static labels (`nsd-pixtral-motion-caption`).

This is a legacy analysis script.
"""

from convergence.nsd import get_resource, get_subject_roi
from convergence.metrics import measure
import matplotlib.pyplot as plt
import torch
from tqdm import trange, tqdm
from dmf.alerts import alert, send_alert
import pandas as pd
import numpy as np


def get_info(kind_object="supercategory"):
    info = get_resource("coco-objects")
    # pivot supercategory to columns nsd_id as index 1 if the supercategory is present 0 otherwise

    info = info.drop_duplicates(subset=["nsd_id", kind_object])
    info = (
        info.pivot(index="nsd_id", columns=kind_object, values=kind_object)
        .notnull()
        .astype(int)
    )
    info = info.reset_index()

    df_motion = pd.read_parquet(
        "/mnt/tecla/Results/convergence/captions/nsd-pixtral-motion-caption.parquet"
    )
    df_motion["motion"] = (df_motion["caption"] == "motion").astype(int)
    df_motion["static"] = (df_motion["caption"] == "static").astype(int)

    df_motion = df_motion[["nsd_id", "motion", "static"]]
    info = info.merge(df_motion, on="nsd_id")
    return info


def get_merge_indexes(subject_x: int, subject_y: int, query: str = ""):
    df = get_resource("stimulus")
    total_stimuli = len(df)
    info = get_info()

    df = df.merge(info, on="nsd_id")
    assert len(df) == total_stimuli

    # Query
    if query:
        df = df.query(query)

    df_x = df.query(f"subject == {subject_x} and exists and shared")
    df_y = df.query(f"subject == {subject_y} and exists and shared")
    df_x = df_x[
        [
            "subject",
            "subject_index",
            "nsd_id",
            "repetition",
            "session",
        ]
    ]
    # Add _x to the column names
    # df_x.columns = [f"{col}_x" for col in df_x.columns]
    df_y = df_y[["subject", "subject_index", "nsd_id", "repetition", "session"]]
    # Add _y to the column names

    df_merge = df_x.merge(
        df_y,
        left_on=["nsd_id", "repetition"],
        right_on=["nsd_id", "repetition"],
    )
    return df_merge


def get_betas_roi(subject: int, roi: int, subject_index, q=0.003):
    betas = get_subject_roi(subject=subject, roi=roi)
    betas = betas[subject_index]
    a, b = np.quantile(betas, [q, 1 - q])
    betas = np.clip(betas, a, b)
    betas = (betas - a) / (b - a)
    betas = torch.tensor(betas, dtype=torch.float32, device="cuda")
    return betas


@alert(input=["output_file", "metrics", "diagonal"], output=True)
def extract_alignment_session_full(
    output_file,
    model_path,
    metrics=["unbiased_cka", "rsa"],
    diagonal=False,
    partitions=[("all", "")],
):
    results = []
    for subject_x in trange(1, 9, position=0, leave=False, desc="Subject X"):
        send_alert(f"Subject X: {subject_x}")
        for subject_y in trange(1, 9, position=1, leave=False, desc="Subject Y"):
            if subject_x == subject_y:
                continue

            for partition, query in tqdm(
                partitions, position=2, leave=False, desc="Partition"
            ):

                df_merge = get_merge_indexes(
                    subject_x=subject_x, subject_y=subject_y, query=query
                )
                subject_index_x = df_merge["subject_index_x"].tolist()
                subject_index_y = df_merge["subject_index_y"].tolist()

                roi_y_cache = {}
                for roi_x in trange(1, 360 + 1, position=3, leave=False, desc="ROI X"):
                    betas_x = get_betas_roi(
                        subject=subject_x, roi=roi_x, subject_index=subject_index_x
                    )

                    for roi_y in trange(
                        1, 360 + 1, position=4, leave=False, desc="ROI Y"
                    ):
                        if diagonal and roi_x != roi_y:
                            continue

                        if diagonal:
                            betas_y = get_betas_roi(
                                subject=subject_y,
                                roi=roi_y,
                                subject_index=subject_index_y,
                            )
                        else:  # Cache the betas
                            if roi_y not in roi_y_cache:
                                roi_y_cache[roi_y] = get_betas_roi(
                                    subject=subject_y,
                                    roi=roi_y,
                                    subject_index=subject_index_y,
                                )
                            betas_y = roi_y_cache[roi_y]

                        for metric in metrics:
                            score = measure(
                                feats_A=betas_x,
                                feats_B=betas_y,
                                metric_name=metric,
                            )
                            results.append(
                                {
                                    "subject_x": subject_x,
                                    "subject_y": subject_y,
                                    "roi_x": roi_x,
                                    "roi_y": roi_y,
                                    "n_stimuli": len(subject_index_x),
                                    "metric": metric,
                                    "score": score,
                                    "partition": partition,
                                }
                            )

    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_file)


if __name__ == "__main__":
    output = "nsd_inter_subject_sessions_partitions.parquet"
    metrics = ["unbiased_cka", "rsa"]
    diagonal = True
    # extract_alignment_session_wise(output_file=output, metrics=metrics, diagonal=diagonal)

    partitions = [
        ("all", ""),
        ("motion", "motion==1"),
        ("static", "static==1"),
        ("person", "person==1"),
        ("non-person", "person==0"),
        ("non-person static", "person==0 and static==1"),
        ("non-person motion", "person==0 and static==0"),
        ("person static", "person==1 and static==1"),
        ("person motion", "person==1 and static==0"),
        ("sports", "sports==1"),
        ("food static", "food==1 and person==0 and static==1"),
        ("motion animal", "animal==1 and person==0 and static == 0"),
        ("static animal", "static==1 and animal==1 and person == 0"),
    ]

    output = "nsd_inter_subject_partitions_separated.parquet"
    extract_alignment_session_full(
        output_file=output, metrics=metrics, diagonal=diagonal, partitions=partitions
    )

"""
Compute cross-participant or subject-to-model alignment scores across stimulus partitions.

This script supports two main modes:
1. **Subject-Subject alignment**: compares beta responses across subjects within NSD,
   matched on shared images (with optional repetition shifts), evaluated over:
   - Full set of shared images
   - Image repetition subsets (rep=0/1/2)
   - Object-based partitions (COCO supercategories and their complements)
   - Random subsets (5x, 50% split)

2. **Subject-Model alignment** *(WIP)*: compares model features to subject ROI responses,
   across image partitions. Requires precomputed model features indexed by `nsd_id`.

Supported metrics:
- RSA
- (Unbiased) CKA
- Mutual KNN
- Linear CKA

Input:
- NSD subject ROI betas
- Stimulus metadata with image/subject mappings
- COCO object annotations (for content-based partitions)

Output:
- A `.parquet` file with pairwise similarity scores for each ROI, metric, and partition

Notes:
- This script is modular and partly extensible to subject-model alignment.
- To run subject-model alignment, implement `compare_session_partitions()`.
"""


import argparse
import numpy as np
import pandas as pd
import torch

from tqdm import trange, tqdm
from dmf.alerts import send_alert, alert

from convergence.metrics import measure
from convergence.nsd import get_subject_roi, get_resource, get_index


def compare(
    betas_i,
    betas_j,
    indexes,
    metric="rsa",
    **kwargs,
):

    results = []
    if indexes is not None:
        betas_i = betas_i[indexes]
        betas_j = betas_j[indexes]

    if isinstance(metric, str):
        metric = [metric]

    for metric_name in metric:
        r = measure(metric_name=metric_name, feats_A=betas_i, feats_B=betas_j)

        result = {
            "n_stim": len(betas_i),
            "metric": metric_name,
            "score": r,
            **kwargs,
        }
        results.append(result)

    return results


def get_nsd_ids(df, categories):
    inclued = df[categories].sum(axis=1) > 0
    return df[inclued].nsd_id.tolist()


def get_common_indexes(df, subject_i, subject_j, shift=0):
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


def get_object_partitions():
    df_objects_detail = get_resource("coco-objects")

    supercategories = df_objects_detail["supercategory"].unique().tolist()

    df_objects = (
        df_objects_detail.groupby(["nsd_id", "supercategory"])
        .aggregate({"area": "sum", "category_id": "count"})
        .reset_index()
    )
    df_objects = df_objects.pivot(
        index="nsd_id", columns="supercategory", values=["category_id"]
    ).reset_index()
    df_objects = df_objects.fillna(0)
    df_objects = df_objects.astype(int)

    columns = [c[-1] if c[-1] else c[0] for c in df_objects.columns.values]
    df_objects.columns = columns

    return df_objects, supercategories


def prepare_betas(betas, indexes, clip=0.003):
    q0, q1 = np.quantile(betas, [clip, 1 - clip])
    betas = betas[indexes]
    betas = np.clip(betas, q0, q1)
    betas = (betas - q0) / (q1 - q0)
    betas = torch.tensor(betas, device="cuda", dtype=torch.float32)
    return betas


def compare_cross_participants(
    subject_i,
    subject_j,
    df_index_common,
    df_objects,
    supercategories,
    metric="rsa",
    shift=0,
):

    results_ij = []
    df_merged = get_common_indexes(
        df_index_common, subject_i=subject_i, subject_j=subject_j, shift=shift
    )

    for roi in trange(1, 361, position=2, desc="ROI", leave=False):
        betas_i = get_subject_roi(subject_i, roi)
        betas_i = prepare_betas(betas_i, df_merged.subject_index_i.tolist())

        betas_j = get_subject_roi(subject_j, roi)
        betas_j = prepare_betas(betas_j, df_merged.subject_index_j.tolist())

        info = {
            "subject_i": subject_i,
            "subject_j": subject_j,
            "roi": roi,
            "metric": metric,
            "shift": shift,
        }

        # Partition with all images in common
        result = compare(
            betas_i=betas_i, betas_j=betas_j, indexes=None, partition_name="all", **info
        )
        results_ij.extend(result)

        # Partitions by repetition
        for rep in range(0, 3):
            indexes = (df_merged.repetition_i == rep).tolist()
            name = f"repetition_{rep}"
            result = compare(
                betas_i=betas_i,
                betas_j=betas_j,
                indexes=indexes,
                partition_name=name,
                **info,
            )
            results_ij.extend(result)

        # Partitions by supercategory
        for supercategory in supercategories:
            nsd_ids = get_nsd_ids(df_objects, [supercategory])
            indexes = df_merged.nsd_id_i.isin(nsd_ids).tolist()
            name = f"{supercategory}"
            result = compare(
                betas_i=betas_i,
                betas_j=betas_j,
                indexes=indexes,
                partition_name=name,
                **info,
            )
            results_ij.extend(result)

            # Complement
            indexes = (~df_merged.nsd_id_i.isin(nsd_ids)).tolist()
            name = f"not_{supercategory}"
            result = compare(
                betas_i=betas_i,
                betas_j=betas_j,
                indexes=indexes,
                partition_name=name,
                **info,
            )
            results_ij.extend(result)

        # Random parititions with 50% of the images
        for n_random in range(5):
            indexes = np.random.choice(df_merged.index, int(len(df_merged) / 2), replace=False)
            name = f"random_{n_random}"
            result = compare(
                betas_i=betas_i,
                betas_j=betas_j,
                indexes=indexes,
                partition_name=name,
                **info,
            )
            results_ij.extend(result)

        del betas_i, betas_j

    torch.cuda.empty_cache()

    return results_ij


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cross-subject similarities with various metrics and configurations."
    )

    # Argument for metric
    parser.add_argument(
        "--metric",
        nargs="+",  # Allows multiple choices
        choices=["rsa", "unbiased_cka", "mutual_knn", "cka"],
        default=["rsa", "unbiased_cka"],  # Default is all metrics
        help="List of metrics to compute. Choices: rsa, unbiased_cka, mutual_knn, cka. Default is all.",
    )

    # Argument for output filename
    parser.add_argument(
        "--output_filename",
        type=str,
        default="subject_model_similarities_partitions.parquet",  # Default will be computed based on shift
        help="Output filename for the results. Default is based on the shift value.",
    )

    # Parse the arguments
    args = parser.parse_args()

    if not args.output_filename.endswith(".parquet"):
        args.output_filename += ".parquet"

    return args


def compare_models(
    subject: int,
    model: torch.Tensor,
    df_index: pd.DataFrame,
    df_objects: pd.DataFrame,
    supercategories: list,
    metric: str,
    model_path: str,
):
    results_ij = []
    df_subject = df_index.query("subject == @subject and exists")

    for roi in trange(1, 361, position=2, desc="ROI", leave=False):
        sessions = list(df_subject.session.unique())
        roi_betas = get_subject_roi(subject=subject, roi=roi).astype(np.float32)

        for session in tqdm(sessions, position=3, desc="Session", leave=False):
            df_session = df_subject.query("session == @session")
            subject_index = df_session.subject_index.tolist()
            nsd_id = df_session.nsd_id.tolist()

            roi_betas_session = prepare_betas(roi_betas[subject_index])
            features_session = model[nsd_id].to("cuda")

            info = {
                "subject": subject,
                "model_path": model_path,
                "roi": roi,
                "metric": metric,
            }

            r = compare_session_partitions(
                roi_betas=roi_betas_session,  # n_stim x n_features
                features=features_session,
                df_objects=df_objects,
                nsd_ids=nsd_id,
                supercategories=supercategories,
                metric=metric,
                **info,
            )
            results_ij.extend(r)

    return results_ij


def compare_session_partitions(
    roi_betas: torch.Tensor,
    features: torch.Tensor,
    df_objects: pd.DataFrame,
    nsd_ids: list,
    supercategories: list,
    metric: str,
    **kwargs,
):

    return []


def main():

    args = parse_args()
    metric = args.metric
    output_filename = args.output_filename

    df_index = get_index("stimulus")
    df_objects, supercategories = get_object_partitions()
    models = []

    try:
        results = []
        for model_path in tqdm(models, position=0, desc="Model", leave=False):
            model_features = torch.load(model_path, weights_only=True)
            model_features = model_features["feats"]
            for subject in trange(1, 8, position=1, leave=False, desc="Subject"):

                results_ij = compare_models(
                    subject=subject,
                    model=model_features,
                    model_path=model_path,
                    df_index=df_index,
                    df_objects=df_objects,
                    supercategories=supercategories,
                    metric=metric,
                )
                results.extend(results_ij)

    except (KeyboardInterrupt, Exception) as e:
        send_alert("Interrupted", str(e), level="error")
        raise e
    finally:
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_parquet(output_filename, index=False)


if __name__ == "__main__":
    main()

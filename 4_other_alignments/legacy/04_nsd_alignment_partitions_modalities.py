"""
Legacy script to compute partitioned RSA alignment across subjects and models.

For each ROI and data partition (e.g. scenes, categories), computes:
- Inter-subject alignment using common NSD stimuli.
- Model-brain alignment per subject and model layer.

Requires precomputed ROI betas and model features. Used to assess how alignment
varies across different content-defined subsets of the stimuli.
"""

import pandas as pd
import argparse
from pathlib import Path
from convergence.nsd import get_resource, get_subject_roi, get_common_indexes
from convergence.metrics import measure
from dmf.alerts import alert, send_alert
from dmf.io import load
import numpy as np
import torch
from tqdm import tqdm, trange
import gc


def prepare_betas(subject, roi, subject_index, separate_hemispheres=False, q=0.003):
    if not separate_hemispheres:
        roi = [roi, roi + 180]
    betas = get_subject_roi(subject=subject, roi=roi)
    betas = betas[subject_index]
    a, b = np.quantile(betas, [q, 1 - q])
    betas = np.clip(betas, a, b)
    betas = (betas - a) / (b - a)
    betas = torch.tensor(betas, dtype=torch.float32).to("cuda")
    return betas


@alert
def compute_intersubject_partition(df_partitions, separate_hemispheres=False):
    results = []
    partitions = df_partitions.columns[1:]
    total_rois = 360 if separate_hemispheres else 180
    metric = "rsa" # unbiased_cka
    for subject_i in trange(1, 9, desc="Subject i", position=0, leave=False):
        for subject_j in trange(1, 9, desc="Subject j", position=1, leave=False):
            if subject_i >= subject_j:
                continue

            df_common_indexes = get_common_indexes(
                subject_i=subject_i, subject_j=subject_j
            )
            df_common_indexes = df_common_indexes.drop(columns="nsd_id_j").rename(
                columns={"nsd_id_i": "nsd_id"}
            )
            df_common_indexes = df_common_indexes[
                df_common_indexes.nsd_id.isin(df_partitions.nsd_id)
            ]
            df_common_indexes = df_common_indexes.merge(df_partitions, on="nsd_id")
            subject_index_i = df_common_indexes.subject_index_i.tolist()
            subject_index_j = df_common_indexes.subject_index_j.tolist()

            for roi in trange(1, total_rois + 1, desc="ROIs", position=2, leave=False):
                betas_i = prepare_betas(
                    subject=subject_i, roi=roi, subject_index=subject_index_i
                )
                betas_j = prepare_betas(
                    subject=subject_j, roi=roi, subject_index=subject_index_j
                )
                for partition in tqdm(
                    partitions, desc="Partitions", position=3, leave=False
                ):
                    partition_values = list(df_common_indexes[partition].unique())
                    for partition_value in partition_values:
                        betas_partition_i = betas_i[
                            df_common_indexes[partition] == partition_value
                        ]
                        betas_partition_j = betas_j[
                            df_common_indexes[partition] == partition_value
                        ]
                        n_stimuli = betas_partition_i.shape[0]
                        
                        cka = measure(metric, betas_partition_i, betas_partition_j)
                        results.append(
                            {
                                "subject_i": subject_i,
                                "subject_j": subject_j,
                                "roi": roi,
                                "metric": metric,
                                "partition": partition,
                                "partition_value": partition_value,
                                "n_stimuli": n_stimuli,
                                "cka": float(cka),
                            }
                        )
    df_results = pd.DataFrame(results)
    # Make subjects symetric
    df_results_symetric = df_results.copy()
    df_results_symetric["subject_i"], df_results_symetric["subject_j"] = (
        df_results["subject_j"],
        df_results["subject_i"],
    )
    df_results = pd.concat([df_results, df_results_symetric]).reset_index(drop=True)
    return df_results


def load_models():
    models_folder = Path("/mnt/tecla/Results/convergence/features/nsd/all")
    models = models_folder.glob("*.pt")
    # Filter oonly thos that ends with -coco.pt or -cls.pt
    models = [
        model
        for model in models
        if model.stem.endswith("-coco") or model.stem.endswith("-cls")
    ]
    # That not contains "bigscience" or "google" in the path
    models = [
        model
        for model in models
        if "bigscience" not in str(model) and "google" not in str(model)
    ]
    return models


def prepare_model_features(features, nsd_ids, q=0.003):
    features = features[nsd_ids]
    a, b = np.quantile(features, [q, 1 - q])
    features = np.clip(features, a, b)
    features = (features - a) / (b - a)
    features = torch.tensor(features, dtype=torch.float32).to("cuda")
    return features

def prepare_subject_cache(features, nsd_ids):
    model_subject_cache = {}
    for layer in range(features.shape[1]):
        model_subject_cache[layer] = prepare_model_features(
            features[:, layer, :], nsd_ids
        )
    return model_subject_cache

def get_partition_values_dict(df_partitions):
    partitions = df_partitions.columns[1:]
    partition_values_dict = {}
    for partition in partitions:
        partition_values_dict[partition] = list(df_partitions[partition].unique())
    return partition_values_dict

def get_subject_masks(df_stimuli, partition_values_dict):
    subject_stimuli_masks = {}
    for partition, partition_values in partition_values_dict.items():
        subject_stimuli_masks[partition] = {}
        for partition_value in partition_values:
            subject_stimuli_masks[partition][partition_value] = (
                df_stimuli[partition] == partition_value
            ).tolist()
    return subject_stimuli_masks

@alert
def compute_models_partition(df_partitions, separate_hemispheres=False):
    models = load_models()
    df_stimuli = get_resource("stimulus").query("shared and exists")
    df_stimuli = df_stimuli.merge(df_partitions, on="nsd_id")
    total_rois = 360 if separate_hemispheres else 180
    partitions = df_partitions.columns[1:]
    partition_values_dict = get_partition_values_dict(df_partitions)
    metric = "rsa" # unbiased_cla
    results = []
    try:
        for model in tqdm(models, desc="Models", position=0):
            send_alert(f"Computing {model.stem}")
            features = torch.load(model, weights_only=True)
            features = features["feats"].to(torch.float32).numpy()
            n_layers = features.shape[1]
            model_name = model.stem

            for subject in trange(1, 9, desc="Subject", position=1, leave=False):
                df_stimuli_subject = df_stimuli.query("subject == @subject")
                nsd_id = df_stimuli_subject.nsd_id.tolist()
                subject_index = df_stimuli_subject.subject_index.tolist()
                model_subject_cache = prepare_subject_cache(features, nsd_id)
                subject_stimuli_masks = get_subject_masks(df_stimuli_subject, partition_values_dict)
                for roi in trange(1, total_rois + 1, desc="ROIs", position=2):
                    betas = prepare_betas(subject=subject, roi=roi, subject_index=subject_index)
                    for layer in trange(n_layers, desc="Layers", position=3, leave=False):
                        features_layer = model_subject_cache[layer]
                        for partition in partitions:
                            partition_values = partition_values_dict[partition]
                            for partition_value in partition_values:
                                partition_mask = subject_stimuli_masks[partition][partition_value]
                                betas_partition = betas[partition_mask]
                                features_partition = features_layer[partition_mask]
                                cka = measure(metric, betas_partition, features_partition)
                                results.append(
                                    {
                                        "model": model_name,
                                        "subject": subject,
                                        "roi": roi,
                                        "layer": layer,
                                        "metric": metric,
                                        "partition": partition,
                                        "partition_value": partition_value,
                                        "n_stimuli": betas_partition.shape[0],
                                        "cka": float(cka),
                                    }
                                )
                del model_subject_cache
                gc.collect()
                torch.cuda.empty_cache()
       
    except (Exception, KeyboardInterrupt) as e:
        if results:
            send_alert("Interrupted")
            df_results = pd.DataFrame(results)
            df_results.to_parquet("nsd_alignment_models_partitions_interrupted.parquet", index=False)
        raise e

    df_results = pd.DataFrame(results)
    return df_results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    # partitions file
    parser.add_argument(
        "--partitions", type=str, default="partitions_subsets_alignment.csv"
    )
    parser.add_argument("--separate-hemispheres", action="store_true")

    parser.add_argument(
        "--output-intersubject",
        type=str,
        default="nsd_alignment_intersubject_partitions_rsa.parquet",
    )
    parser.add_argument(
        "--output-models", type=str, default="nsd_alignment_models_partitions_rsa.parquet"
    )

    args = parser.parse_args()
    partitions = args.partitions
    separate_hemispheres = args.separate_hemispheres

    output_intersubject = Path(args.output_intersubject)
    output_models = Path(args.output_models)

    # Read partitions
    df_partitions = load(partitions)
    df_partitions["all_scenes"] = "all"

    send_alert("Computing intersubject partition")
    df_subjects = compute_intersubject_partition(df_partitions, separate_hemispheres)
    df_subjects.to_parquet(output_intersubject, index=False)

    send_alert("Computing models partition")
    df_models = compute_models_partition(df_partitions, separate_hemispheres)
    df_models.to_parquet(output_models, index=False)


if __name__ == "__main__":
    main()

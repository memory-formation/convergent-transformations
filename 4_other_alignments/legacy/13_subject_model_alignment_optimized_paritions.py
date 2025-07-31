"""
Compute subject–model alignment scores across multiple partitions of NSD stimuli.

This script:
- Computes similarity (RSA, CKA, etc.) between model features and subject ROI responses.
- Evaluates alignment across several semantic and motion-based partitions (e.g., person, static, sports).
- Iterates over all ROIs (1–360), 8 subjects, and multiple models.
- For each ROI and partition, compares subject betas to model features across all layers.

Input:
- NSD beta activations (via `get_subject_roi`)
- Precomputed model features (tensor with shape [n_stimuli, n_layers, n_features])

Partitions:
- Includes predefined categories (e.g., person, motion, animal, etc.) and their intersections.

Output:
- A `.parquet` file with one row per model, subject, ROI, layer, metric, and partition.

Note:
- Clipping and normalization are applied to both model and brain features.
- Uses GPU for efficient computation.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import trange, tqdm
from dmf.alerts import send_alert, alert
import gc

from convergence.metrics import measure
from convergence.nsd import get_subject_roi, get_resource, get_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cross-subject similarities with various metrics and configurations."
    )

    # Argument for metric
    parser.add_argument(
        "--metric",
        nargs="+",  # Allows multiple choices
        choices=["rsa", "unbiased_cka", "mutual_knn", "cka"],
        default=["unbiased_cka"],  # Default is all metrics
        help="List of metrics to compute. Choices: rsa, unbiased_cka, mutual_knn, cka. Default is all.",
    )

    # Argument for output filename
    parser.add_argument(
        "--output_filename",
        type=str,
        default="subject_model_similarities.parquet",  # Default will be computed based on shift
        help="Output filename for the results. Default is based on the shift value.",
    )

    # Parse the arguments
    args = parser.parse_args()

    if not args.output_filename.endswith(".parquet"):
        args.output_filename += ".parquet"

    return args

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

def prepare_betas(betas, clip=0.003):
    q0, q1 = np.quantile(betas, [clip, 1 - clip])
    betas = np.clip(betas, q0, q1)
    betas = (betas - q0) / (q1 - q0)
    betas = torch.tensor(betas, device="cuda", dtype=torch.float32)
    return betas

def prepare_features(features, q=0.997):

    # Shape n_stim x n_layers x n_features
    # For each layers, apply clipping to quantile 0.003. Similar to betas


    features = features.to(torch.float32).to("cuda")
    for n_layer in range(features.shape[1]):

        layer = features[:, n_layer]
        q_val = layer.reshape(-1).abs().sort().values[int(q * layer.numel())]
        layer.clamp_(0, q_val)
        min_val = layer.min()
        max_val = layer.max()
        layer.sub_(min_val).div_(max_val - min_val)
        features[:, n_layer] = layer
        
    return features

def get_partitions_index(subject, query):
    df = get_resource("stimulus")
    total_stimuli = len(df)
    info = get_info()

    df = df.merge(info, on="nsd_id")
    assert len(df) == total_stimuli

    # Query
    if query:
        df = df.query(query)

    df_subject = df.query(f"subject == {subject} and exists and shared")
    return df_subject

@alert(input=["subject", "metrics"])
def compare_models(
    subject: int,
    model: torch.Tensor,
    metrics: str,
    partitions: list,
    **kwargs,
):
    results_ij = []
  

    cache_model_session = {}

    for roi in trange(1, 361, position=2, desc="ROI", leave=False):

        roi_betas = get_subject_roi(subject=subject, roi=roi).astype(np.float32)

        for session, query in tqdm(partitions, position=3, desc="Partition", leave=False):

            df_session = get_partitions_index(subject=subject, query=query)

            subject_index = df_session.subject_index.tolist()
            nsd_id = df_session.nsd_id.tolist()

            roi_betas_session = prepare_betas(roi_betas[subject_index])

            if session not in cache_model_session:
                cache_model_session[session] = prepare_features(model[nsd_id])

            features_session = cache_model_session[session]

            for metric in metrics:
                info = {
                    "subject": subject,
                    "roi": roi,
                    "metric": metric,
                    "partition": session,
                    "n_images": len(df_session),
                    **kwargs,
                }

                r = compare_session(
                    betas=roi_betas_session, features=features_session, **info
                )
                results_ij.extend(r)

    del cache_model_session, features_session, roi_betas, roi_betas_session
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return results_ij


def compare_session(betas, features, metric, **kwargs):

    results = []
    for layer in range(features.shape[1]):
        features_layer = features[:, layer]
        score = measure(metric_name=metric, feats_A=betas, feats_B=features_layer)
        result = {
            "layer": layer,
            "metric": metric,
            "score": score,
            **kwargs,
        }
        results.append(result)

    return results

@alert
def main():

    args = parse_args()
    metrics = args.metric
    output_filename = args.output_filename

    models_folder = Path("/mnt/tecla/Results/convergence/features/nsd/all")
    models = list(models_folder.glob("*.pt"))
    # Filter those that end with -pixtral.pt
    models = [model for model in models if not model.stem.endswith("-pixtral")]
    # Model does not contain word google or bloomz
    models = [model for model in models if "google" not in model.stem and "bloomz" not in model.stem]

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
    
    try:
        results = []
        for model_path in tqdm(models, position=0, desc="Model", leave=False):
            send_alert(model_path.stem)
            model_features = torch.load(model_path, weights_only=True)
            model_features = model_features["feats"]
            for subject in trange(1, 9, position=1, leave=False, desc="Subject"):

                results_ij = compare_models(
                    subject=subject,
                    model=model_features,
                    metrics=metrics,
                    model_name=model_path.stem,
                    partitions=partitions,
                )
                results.extend(results_ij)
                

    except (KeyboardInterrupt, Exception) as e:
        send_alert("Interrupted"+ str(e), level="error")
        raise e
    finally:
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_parquet(output_filename, index=False)
    df_results.to_parquet("safety.parquet", index=False)

if __name__ == "__main__":
    main()

"""
subject_model_alignment_other_metrics.py

This script computes alignment scores between brain data (subject-level ROI responses)
and model features across multiple similarity metrics.

It generalizes the standard subject-to-model RSA framework by supporting multiple
comparison metrics, including:
- RSA (Representational Similarity Analysis)
- CKA (Centered Kernel Alignment)
- Unbiased CKA
- Mutual k-Nearest Neighbors (kNN) similarity

The script loops over subjects and brain ROIs, compares their betas to each model layer’s
features using the selected metric(s), and aggregates results into a DataFrame.

Usage:
    Run from the command line with optional arguments:

    python subject_model_alignment_optimized.py --metric rsa unbiased_cka --output_filename results.parquet

Arguments:
    --metric             List of similarity metrics to compute. Options:
                         ['rsa', 'cka', 'unbiased_cka', 'mutual_knn']
    --output_filename    Path to output `.parquet` file. Defaults to `subject_model_similarities.parquet`.

Requirements:
    - Precomputed model features (.pt files) in: `/mnt/tecla/Results/convergence/features/nsd/all/`
    - Brain ROI betas available via `convergence.nsd.get_subject_roi`
    - Metadata index accessible via `get_resource("stimulus")`

Outputs:
    - A `.parquet` file with alignment scores across layers, ROIs, subjects, sessions, and models.

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

@alert(input=["subject", "metrics"])
def compare_models(
    subject: int,
    model: torch.Tensor,
    metrics: str,
    **kwargs,
):
    results_ij = []
    df_index = get_resource("stimulus")
    df_subject = df_index.query("subject == @subject and exists")
    sessions = list(df_subject.session.unique())

    cache_model_session = {}

    for roi in trange(1, 180+1, position=2, desc="ROI", leave=False):

        roi_betas = get_subject_roi(subject=subject, roi=[roi, roi+180]).astype(np.float32)

        for session in tqdm(sessions, position=3, desc="Session", leave=False):

            df_session = df_subject.query("session == @session")

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
                    "session": session,
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

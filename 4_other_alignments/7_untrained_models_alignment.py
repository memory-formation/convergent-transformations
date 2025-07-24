"""
untrained_models_alignment.py

This script evaluates how *untrained* models (vision and language) align with human brain responses
from the NSD dataset. It computes representational similarity between model features and fMRI betas
across subjects, ROIs, and sessions using a chosen metric (default: `unbiased_cka`).

### Purpose
To assess whether untrained models encode useful structure aligned with brain representations, 
acting as a baseline comparison for trained models.

### Workflow
1. Load untrained model features (from `/features/nsd/all-untrained/`).
2. For each model:
    - Loop through all 8 NSD subjects.
    - For each ROI (joined or separate), and each session:
        - Extract betas.
        - Extract model features for the presented stimuli (nsd_ids).
        - Compute alignment using selected metric (e.g., CKA).
3. Save results to `untrained_vision_similarity.parquet` or `untrained_language_similarity.parquet`.

### Metrics Supported
- `unbiased_cka` (default)
- Extendable to other metrics via `convergence.metrics.measure`

### Key Functions
- `compute_model_brain_similarity`: Main loop over subjects and ROIs
- `compute_rois_similarity`: Per-ROI, per-session alignment computation
- `compute_similarity`: Compare each model layer with beta signals
- `prepare_betas`: Normalizes and clips beta data for comparison

### Inputs
- Model features (e.g., `vit-b-32-untrained.pt`, `clip-coco-untrained.pt`)
- NSD betas and stimulus metadata
- Optional: HCP ROI labels for later plotting

### Outputs
- `untrained_vision_similarity.parquet`
- `untrained_language_similarity.parquet`
Each row: subject, ROI, model, layer, session, score, metric

### Environment
Requires the environment variable:
```bash
export CONVERGENCE_RESULTS=/path/to/results
```
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from convergence.metrics import measure
from convergence.nsd import get_subject_roi, get_resource, get_index
from convergence.plotting import plot_faverage_parcelation
from tqdm import trange, tqdm
from dmf.alerts import send_alert, alert


def get_indexes_subject(df_simuli: pd.DataFrame, subject: int, session=None):
    query = "exists and subject == @subject"
    if session is not None:
        query += " and session == @session"
    df_session = df_simuli.query(query)
    nsd_id = df_session["nsd_id"].tolist()
    subject_index = df_session["subject_index"].tolist()

    return nsd_id, subject_index


def prepare_betas(betas: np.ndarray, device="cuda", q=0.003) -> torch.Tensor:
    betas = betas.astype(np.float32)
    if q:
        q0, q1 = np.quantile(betas.ravel(), [q, 1 - q])
        betas = np.clip(betas, q0, q1, out=betas)
        betas -= q0
        betas /= q1 - q0
    betas = torch.tensor(betas, dtype=torch.float32, device=device)
    return betas


def compute_similarity(betas, features, metric="unbiased_cka", **kwargs):
    results = []
    if isinstance(metric, str):
        metric = [metric]
    n_stim = len(betas)
    for layer in range(features.shape[1]):
        layer_features = features[:, layer, :]
        for m in metric:
            r = measure(m, betas, layer_features)
            results.append(
                dict(layer=layer, metric=m, score=r, n_stim=n_stim, **kwargs)
            )
    return results


def compute_rois_similarity(
    features=None,
    subject=1,
    sessions=None,
    df_stimuli=None,
    q=None,
    metric="unbiased_cka",
    join_hemisphere=True,
    **kwargs,
):
    if df_stimuli is None:
        df_stimuli = get_index("stimulus")

    results = []
    if sessions is None:
        sessions = df_stimuli.query("subject == @subject and exists")[
            "session"
        ].unique()

    if join_hemisphere:
        total_rois = 180
    else:
        total_rois = 360

    for roi in trange(1, total_rois + 1, leave=False):
        if join_hemisphere:
            betas_roi = get_subject_roi(subject=subject, roi=[roi, roi + 180])
        else:
            betas_roi = get_subject_roi(subject=subject, roi=roi)
        #betas_roi = prepare_betas(betas_roi, q=q)
        for session in tqdm(sessions,  leave=False, position=3):
            nsd_id, subject_index = get_indexes_subject(
                df_stimuli, subject=subject, session=session
            )
            betas_session = betas_roi[subject_index]
            betas_session = prepare_betas(betas_session, q=q)
            features_session = features[nsd_id].to("cuda")
            session_result = compute_similarity(
                betas=betas_session,
                features=features_session,
                metric=metric,
                roi=roi,
                subject=subject,
                session=session,
                **kwargs,
            )
            results.extend(session_result)
    return results


def untrained_path_to_trained(path):
    return Path(str(path).replace("-untrained", "").replace("all-untrained", "all"))

@alert
def compute_model_brain_similarity(
    model_path, subject=None, sessions=None, q=0.003, metric="unbiased_cka", join_hemisphere=True, **kwargs
):
    features = torch.load(model_path, weights_only=True)["feats"].to(torch.float32)
    model_name = Path(model_path).stem

    if subject is None:
        subject = list(range(1, 9))
    elif not isinstance(subject, list):
        subject = [subject]

    results = []
    for s in tqdm(subject, leave=False):
        similarities = compute_rois_similarity(
            features=features,
            subject=s,
            sessions=sessions,
            q=q,
            metric=metric,
            model=model_name,
            join_hemisphere=join_hemisphere,
            **kwargs,
        )
        results.extend(similarities)
        torch.cuda.empty_cache()
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":

    results_folder = Path(os.getenv("CONVERGENCE_RESULTS"))
    features_folder = results_folder / "features/nsd/all-untrained"
    feature_paths = list(features_folder.glob("*-untrained.pt"))
    join_hemisphere = True

    df_stimuli = get_index("stimulus")
    df_hcp = get_resource("hcp")
    df_hcp = df_hcp[["roi", "name", "mne_name"]]

    vision_models = [
        feature_path
        for feature_path in feature_paths
        if feature_path.stem.startswith("vit")
    ]
    language_models = [
        feature_path for feature_path in feature_paths if "coco" in feature_path.stem
    ]
    language_models = [
        feature_path
        for feature_path in language_models
        if "bloomz" not in feature_path.stem
    ]
    language_models = [
        feature_path
        for feature_path in language_models
        if "gemma" not in feature_path.stem
    ]

    # Compute similarity for vision models
    data = []
    for model in tqdm(vision_models):
        send_alert(f"Computing similarity for {model.stem}")
        df_similarity = compute_model_brain_similarity(
            model, metric="unbiased_cka", q=0.003, join_hemisphere=join_hemisphere,
        )
        data.append(df_similarity)

    df_vision = pd.concat(data)
    df_vision["modality"] = "vision"
    df_vision.to_parquet("untrained_vision_similarity.parquet")

    send_alert(f"Finished similarity for vision models")

    # Compute similarity for language models
    data_language = []
    for model in language_models:
        send_alert(f"Computing similarity for {model.stem}")
        df_similarity = compute_model_brain_similarity(
            model, metric="unbiased_cka", q=0.003, join_hemisphere=join_hemisphere,
        )
        data_language.append(df_similarity)

    df_language = pd.concat(data_language)
    df_language["modality"] = "language"
    df_language.to_parquet("untrained_language_similarity.parquet")

    send_alert(f"Finished similarity for language models")

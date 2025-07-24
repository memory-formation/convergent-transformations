"""
Performs permutation-based CKA alignment analysis between NSD subject ROIs and deep model features.

For each subject, ROI, and model layer:
- Computes unbiased CKA between fMRI betas and model features.
- Constructs null distributions via within-subject permutation.
- Supports full-layer or top-layer testing.
- Allows session-wise computation and hemisphere separation.

Requires: 
- NSD subject betas (via `get_subject_roi`)
- Precomputed model features (torch .pt format)
- Convergence-compatible structure and alert system

Outputs one parquet file per model under `permutations/`.
"""

import argparse
from pathlib import Path

from tqdm import trange, tqdm
import pandas as pd
import numpy as np
import torch

from convergence.nsd import get_subject_roi, get_resource
from convergence.metrics.utils import hsic_unbiased
from dmf.alerts import alert, send_alert
import gc


def prepare_features(features, q=0.003):
    a, b = np.quantile(features.ravel(), [q, 1 - q])
    features = np.clip(features, a, b)
    features = torch.tensor(features, device="cuda", dtype=torch.float32)
    features = (features - a) / (b - a)

    K = torch.mm(features, features.T)
    return K


@alert(input=True)
def compute_model_permutations_all_sessions(
    model_path,
    n_repetitions=1000,
    join_hemispheres=True,
    n_subjects=8,
    max_sessions=1,
):
    model_features = (
        torch.load(model_path, weights_only=True)["feats"].to(torch.float32).numpy()
    )  # n_stimuli x n_layers x n_features
    n_layers = model_features.shape[1]
    model_name = model_path.stem
    df_stimulus = get_resource("stimulus")
    total_rois = 180 if join_hemispheres else 360
    model_results = []

    for subject_i in trange(1, n_subjects + 1, position=1, leave=False, desc="Subject"):
        df_subject = df_stimulus.query(f"subject == {subject_i} and exists")
        sessions = list(df_subject["session"].unique())
        # Sort the sessions
        sessions = sorted(sessions)
        if max_sessions:
            sessions = sessions[:max_sessions]
        subject_cache_model = {}

        for roi in trange(1, total_rois + 1, position=2, leave=False, desc="ROI"):
            roi_key = roi if not join_hemispheres else [roi, roi + 180]
            betas = get_subject_roi(subject=subject_i, roi=roi_key)

            for session in tqdm(sessions, position=3, leave=False, desc="Session"):
                df_session = df_subject.query(f"session == {session}")
                subject_index = df_session["subject_index"].tolist()
                nsd_id = df_session["nsd_id"].tolist()

                # Get betas for the session
                K = prepare_features(betas[subject_index])  # K
                hsic_kk = hsic_unbiased(K, K)

                # Check if the session is in the cache for not repeating the same computation for layers
                if session not in subject_cache_model:
                    subject_cache_model[session] = {}

                for layer in trange(n_layers, position=4, leave=False, desc="Layer"):
                    if layer not in subject_cache_model[session]:
                        L = prepare_features(model_features[nsd_id, layer, :])  # L
                        hsic_ll = hsic_unbiased(L, L)
                        subject_cache_model[session][layer] = (L, hsic_ll)

                    L, hsic_ll = subject_cache_model[session][layer]  # L
                    normalization_term = torch.sqrt(hsic_kk * hsic_ll) + 1e-6

                    for permutation in trange(
                        n_repetitions + 1, position=5, leave=False, desc="Permutation"
                    ):

                        if permutation == 0:
                            # Original values
                            hsic_kl = hsic_unbiased(K, L)
                        else:
                            # Permute the labels
                            idx = torch.randperm(L.shape[0])
                            hsic_kl = hsic_unbiased(K, L[idx, :][:, idx])

                        cka = hsic_kl / normalization_term
                        model_results.append(
                            {
                                "model": model_name,
                                "subject": subject_i,
                                "session": session,
                                "roi": roi,
                                "layer": layer,
                                "permutation": permutation,
                                "cka": float(cka.item()),
                            }
                        )
        del subject_cache_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        model_results_checkpoint = pd.DataFrame(model_results)
        model_results_checkpoint.to_parquet(
            Path(f"{model_path.stem}_checkpoint.parquet")
        )

    return model_results


@alert(input=True)
def compute_model_permutations(
    model_path,
    n_repetitions=1000,
    join_hemispheres=True,
    n_subjects=8,
    only_top_layers=False,
    df_model_top_layers=None,
):
    model_features = (
        torch.load(model_path, weights_only=True)["feats"].to(torch.float32).numpy()
    )  # n_stimuli x n_layers x n_features
    n_layers = model_features.shape[1]
    model_name = model_path.stem
    df_stimulus = get_resource("stimulus")
    total_rois = 180 if join_hemispheres else 360
    model_results = []

    for subject_i in trange(5, n_subjects + 1, position=1, leave=False, desc="Subject"):
        df_subject = df_stimulus.query(f"subject == {subject_i} and exists and shared")
        subject_cache_model = {}

        for roi in trange(1, total_rois + 1, position=2, leave=False, desc="ROI"):
            # compute_permutations(
            #     model_paths=model_paths,
            #     n_repetitions=args.n_reps,
            #     output=args.output,
            #     join_hemispheres=not args.separated_hemispheres,
            #     only_top_layers=args.only_top_layers,
            # )

            roi_key = roi if not join_hemispheres else [roi, roi + 180]
            betas = get_subject_roi(subject=subject_i, roi=roi_key)

            subject_index = df_subject["subject_index"].tolist()
            nsd_id = df_subject["nsd_id"].tolist()

            # Get betas for the session
            K = prepare_features(betas[subject_index])  # K
            hsic_kk = hsic_unbiased(K, K)

            if only_top_layers:

                top_layer = df_model_top_layers.query(
                    f"subject == {subject_i} and roi == {roi}"
                )
                assert (
                    len(top_layer) == 1
                ), f"Top layer not found for subject {subject_i} and roi {roi}"
                top_layer = top_layer["layer"].values[0]
            else:
                top_layer = None

            for layer in trange(n_layers, position=3, leave=False, desc="Layer"):
                if only_top_layers and layer != top_layer:
                    continue
                if layer not in subject_cache_model:
                    L = prepare_features(model_features[nsd_id, layer, :])  # L
                    hsic_ll = hsic_unbiased(L, L)
                    subject_cache_model[layer] = (L, hsic_ll)

                L, hsic_ll = subject_cache_model[layer]  # L
                normalization_term = torch.sqrt(hsic_kk * hsic_ll) + 1e-6

                for permutation in trange(
                    n_repetitions + 1, position=4, leave=False, desc="Permutation"
                ):

                    if permutation == 0:
                        # Original values
                        hsic_kl = hsic_unbiased(K, L)
                    else:
                        # Permute the labels
                        idx = torch.randperm(L.shape[0])
                        hsic_kl = hsic_unbiased(K, L[idx, :][:, idx])

                    cka = hsic_kl / normalization_term
                    model_results.append(
                        {
                            "model": model_name,
                            "subject": subject_i,
                            "roi": roi,
                            "layer": layer,
                            "permutation": permutation,
                            "cka": float(cka.item()),
                        }
                    )
        del subject_cache_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return model_results


@alert(output=True)
def compute_permutations(
    model_paths,
    n_repetitions=1000,
    output="nsd_cka_cross_subject_permutations.parquet",
    join_hemispheres=True,
    only_top_layers=False,
    session_wise=True,
):
    

    model_paths = list(model_paths)

    df_models = pd.read_parquet(
        "/mnt/tecla/Results/convergence/alignments/subject_model_similarities_cka_joined.parquet"
    )
    df_models = df_models.query(
        "metric == 'unbiased_cka' and score < 1 and score > -0.2"
    )
    df_models = (
        df_models.groupby(["model_name", "subject", "roi", "layer"])
        .score.mean()
        .reset_index()
    )
    df_models = df_models.sort_values(
        ["model_name", "subject", "roi", "score"], ascending=[True, True, True, False]
    )
    df_models = df_models.drop_duplicates(["model_name", "subject", "roi"])
    df_models = df_models.reset_index(drop=True)

    folder = Path("permutations")
    try:
        for model in tqdm(model_paths, position=0, leave=False, desc="Model"):
            model_results_file = folder / Path(f"{model.stem}_results.parquet")
            if model_results_file.exists():
                
                continue
            results = []
            send_alert(f"Processing model {model.stem}")
            if only_top_layers:
                df_model_top_layers = df_models.query(f"model_name == '{model.stem}'")
                assert (
                    len(df_model_top_layers) > 0
                ), f"Model {model.stem} not found in the database"
            else:
                df_model_top_layers = None
            if session_wise:
                res_model = compute_model_permutations_all_sessions(
                    model_path=model,
                    n_repetitions=n_repetitions,
                    join_hemispheres=join_hemispheres,
                )
            else:
                res_model = compute_model_permutations(
                    model,
                    n_repetitions,
                    join_hemispheres=join_hemispheres,
                    only_top_layers=only_top_layers,
                    df_model_top_layers=df_model_top_layers,
                )
            df = pd.DataFrame(res_model)
            df.to_parquet(model_results_file)
            del df, res_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (KeyboardInterrupt, Exception) as e:
        if len(results):
            df_error = pd.DataFrame(results)
            df_error.to_parquet(Path(output).with_suffix(".error.parquet"))
        raise e

    # df = pd.DataFrame(results)
    # df.to_parquet(output)


def main():
    argparser = argparse.ArgumentParser()
    # N-repetitions
    argparser.add_argument("--n_reps", type=int, default=2000)
    argparser.add_argument(
        "--output",
        type=str,
    )
    argparser.add_argument("--only_top_layers", action="store_true")
    argparser.add_argument("--separated_hemispheres", action="store_true")
    argparser.add_argument("--session_wise", action="store_true")
    args = argparser.parse_args()

    model_features = Path("/mnt/tecla/Results/convergence/features/nsd/all").glob(
        "*.pt"
    )

    # All that not contain "google" or "pixtral" or "bloomz"
    model_paths = list(
        filter(
            lambda x: all([y not in str(x) for y in ["google", "-pixtral", "bloomz"]]),
            model_features,
        )
    )
    # Start with vit_base_patch16_clip_224

    #model_name = "meta-llama_Meta-Llama-3-8B_pool"  # "vit_base_patch16_clip_224.laion2b_pool-cls.pt"
    #model_name = "vit_giant_patch14_dinov2.lvd142m_pool-cls"
    #model_paths = list(filter(lambda x: model_name not in str(x), model_paths))
    #print(model_paths)
    # print(model_paths)
    compute_permutations(
        model_paths=model_paths,
        n_repetitions=args.n_reps,
        output=args.output,
        join_hemispheres=not args.separated_hemispheres,
        only_top_layers=args.only_top_layers,
        session_wise=args.session_wise,
    )


if __name__ == "__main__":
    main()

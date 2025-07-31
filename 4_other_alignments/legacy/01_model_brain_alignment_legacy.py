"""
model_brain_alignment_legacy.py

This script performs **voxelwise model-brain alignment** across multiple NSD subjects, sessions,
ROIs, and pretrained model representations. It is an earlier implementation of the core subject-model
alignment analysis, using an exhaustive loop-based structure.

### Functionality

For each subject:
- For each session:
  - Load the betas (BOLD responses) for that session
  - For each model:
    - Load features and align to the brain using different metrics (e.g., RSA, CKA)
    - For each atlas (e.g., HCP, MTL, thalamus):
      - For each ROI in the atlas:
        - Extract voxel-level betas
        - Compute alignment between betas and model features

### Output
For each subject, stores alignment scores in:

### Supported metrics
- `'rsa'`: Pearson correlation over RDMs
- `'unbiased_cka'`: Bias-corrected CKA
- (commented) `'mutual_knn'` with configurable `topk`

### Notes
- Features and betas are aligned based on stimulus identity within each session
- This script assumes all model `.pt` files are stored in a `results/` folder
- Designed for GPU execution with memory cleanup between iterations

### Status
This script is **superseded** by newer, more modular code but can still serve as a reference
or backup method.
"""

from tqdm import tqdm
from pathlib import Path
import gc
import numpy as np
import torch
import pandas as pd

from dmf.alerts import alert, send_message
from convergence.alignment import compute_alignment
from convergence.nsd import get_session_indexes, load_betas, get_index, load_mask


@alert(input=False, output=False)
def compute_alignment_nsd(
    subjects: int,
    model_paths: list[Path],
    metrics: list[tuple[str, dict]],
    atlases: list[str],
    device: torch.device,
):
    for subject in (pbar1 := tqdm(subjects, leave=False, position=0)):
        pbar1.set_description(f"Subject {subject}")
        compute_subject_alignment_nsd(
            subject=subject,
            model_paths=model_paths,
            metrics=metrics,
            atlases=atlases,
            device=device,
        )

    return


@alert(input=["subject", "session"], output=False)
def compute_subject_alignment_nsd(subject, model_paths, metrics, atlases, device):
    subject_scores = []
    df_stimulus = get_index("stimulus")
    sessions = df_stimulus.query("exists and subject==@subject").session.unique()
    for session in (pbar2 := tqdm(sessions, leave=False, position=1)):
        pbar2.set_description(f"Session {session}")
        image_indexes = get_session_indexes(subject=subject, session=session)
        betas = load_betas(subject=subject, session=session)
        for model in (pbar3 := tqdm(model_paths, leave=False, position=2)):
            pbar3.set_description(f"Model {model.stem}")
            model_features = torch.load(model, weights_only=True)
            # Prepare features before selecting the image indexes
            # feats_all = prepare_features(model_features["feats"].float(), exact=True)
            feats = model_features["feats"][image_indexes].float().to(device)
            del model_features  # Free up memory
            gc.collect()
            for atlas in (pbar4 := tqdm(atlases, leave=False, position=3)):
                pbar4.set_description(f"Atlas {atlas}")
                mask = load_mask(subject=subject, roi=atlas)
                rois = np.unique(mask)
                rois = rois[rois > 0]
                for roi in tqdm(rois, desc="ROI", leave=False, position=4):
                    masked_betas = betas[mask == roi, :]
                    masked_betas = masked_betas.T  # N_stimulus x N_voxels
                    masked_betas = masked_betas[:, np.newaxis, :]  # N_stimulus x 1 x N_voxels

                    masked_betas = torch.tensor(masked_betas).float().to(device)
                    for metric, kwargs in metrics:

                        info = {
                            "model": model.stem,
                            "model_path": str(model),
                            "atlas": atlas,
                            "roi": roi,
                            "subject": subject,
                            "session": session,
                            "metric": metric,
                            **kwargs,
                        }
                        try:
                            with torch.no_grad():
                                scores = compute_alignment(
                                    x_feats=masked_betas,
                                    y_feats=feats,
                                    include_all_x=False,
                                    metric=metric,
                                    prepare=False,
                                    **kwargs,
                                )
                            add_score_info(scores, info)
                            subject_scores.extend(scores)
                        except KeyboardInterrupt as e:
                            raise e
                        except Exception as e:
                            print(f"Error: {e}")
                            send_message(f"Error computing {info}")
                            continue

                    # Unload the masked betas
                    masked_betas = masked_betas.cpu()
                    del masked_betas
            # Unload the features
            feats = feats.cpu()

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

            del feats
        # Remove the betas from memory
        del betas

    df = pd.DataFrame(subject_scores)
    output_filename = Path(f"results/alignment_nsd_subject_{subject}.parquet")
    df.to_parquet(output_filename, index=False)

    return output_filename


def add_score_info(scores, info):
    for score in scores:
        score.update(info)


def main():
    model_paths = list(Path("results").glob("**/*pt"))

    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    atlases = [
        "rh.thalamus",
        "lh.thalamus",
        "thalamus",
        "rh.MTL",
        "lh.MTL",
        "MTL",
        "HCP_MMP1",
        "lh.HCP_MMP1",
        "rh.HCP_MMP1",
    ]
    metrics = [
        ("rsa", {}),
        ("unbiased_cka", {}),
        # ("mutual_knn", {"topk": 5}),
        # ("mutual_knn", {"topk": 10}),
        # ("mutual_knn", {"topk": 30}),
        # ("mutual_knn", {"topk": 50}),
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_alignment_nsd(
        subjects=subjects,
        model_paths=model_paths,
        metrics=metrics,
        atlases=atlases,
        device=device,
    )


if __name__ == "__main__":
    main()

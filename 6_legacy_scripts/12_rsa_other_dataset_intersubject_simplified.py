"""
Compute inter-subject RSA for BOLD5000 and THINGS datasets.

For each pair of subjects, the script:
- Matches repeated presentations of the same stimulus (`bold_id` or `things_id`)
- Optionally separates data by session (`session_wise=True`)
- Computes ROI-level RSA between subject pairs, with optional hemisphere merging
- Outputs one `.parquet` file per dataset and configuration

Options:
- Session-wise vs full-pool comparison
- Separate vs joined hemispheres (180 vs 360 ROIs)
- Optional quantile clipping of beta values

Results are saved under `rsa_other_dataset/` as parquet files.
"""

import pandas as pd
import torch
from tqdm import tqdm, trange
from pathlib import Path

from convergence.bold5000 import (
    get_resource as get_bold5000_resource,
    get_betas_roi as get_betas_roi_bold5000,
)
from convergence.things import (
    get_resource as get_things_resource,
    get_betas_roi as get_betas_roi_things,
)
from convergence.metrics import measure


QUANTILE = 0


def measure_rsa(
    df_merge,
    subject_i,
    subject_j,
    q=0.003,
    session_wise=True,
    dataset="bold5000",
    join_hemispheres=False,
    position=0,
):
    """
    Measure the RSA between two subjects' things_id
    """
    results = []
    if not session_wise:
        df_merge = df_merge.copy()
        df_merge["session_i"] = 0
        df_merge["session_j"] = 0

    sessions = df_merge["session_i"].unique()
    total_rois = 180 if join_hemispheres else 360
    for roi in trange(1, total_rois + 1, leave=False, position=position):
        symetric_rois = [roi, roi + 180] if join_hemispheres else [roi]
        if dataset == "bold5000":
            betas_i = get_betas_roi_bold5000(subject_i, roi=symetric_rois)
            betas_j = get_betas_roi_bold5000(subject_j, roi=symetric_rois)
        elif dataset == "things":
            betas_i = get_betas_roi_things(subject_i, roi=symetric_rois)
            betas_j = get_betas_roi_things(subject_j, roi=symetric_rois)

        # Send betas to cuda
        betas_i = torch.tensor(betas_i, dtype=torch.float32, device="cuda")
        betas_j = torch.tensor(betas_j, dtype=torch.float32, device="cuda")
        for session in sessions:
            df_merge_session = df_merge.query("session_i == @session")
            subject_indexes_i = df_merge_session["subject_index_i"].tolist()
            subject_indexes_j = df_merge_session["subject_index_j"].tolist()
            betas_i_session = betas_i[subject_indexes_i]
            betas_j_session = betas_j[subject_indexes_j]

            if q > 0:  # Compute quantile
                a, b = torch.quantile(
                    betas_i_session, torch.tensor([q, 1 - q], device="cuda")
                )
                betas_i_session = torch.clip(betas_i_session, a, b)
                a, b = torch.quantile(
                    betas_i_session, torch.tensor([q, 1 - q], device="cuda")
                )
                betas_j_session = torch.clip(betas_j_session, a, b)

            rsa = measure("rsa", betas_i_session, betas_j_session)
            results.append(
                {
                    "roi": roi,
                    "similarity": rsa,
                    "subject_i": subject_i,
                    "subject_j": subject_j,
                    "session": session,
                    "n_elements": len(subject_indexes_i),
                    "q": q,
                    "dataset": dataset,
                    "join_hemispheres": join_hemispheres,
                }
            )
            del betas_i_session, betas_j_session
        del betas_i, betas_j
        torch.cuda.empty_cache()
    df_results = pd.DataFrame(results)
    return df_results


def compute_bold5000_rsa(session_wise=True, quantile=0.003, folder=None):
    df_bold = get_bold5000_resource("stimulus")
    df_bold = df_bold[
        [
            "bold_id",
            "subject",
            "session",
            "run",
            "subject_index",
            "repetition",
            "stim_source",
        ]
    ]

    subjects = df_bold["subject"].unique()
    stim_sources = [None, *df_bold["stim_source"].unique()]
    for join_hemispheres in [False, True]:
        results = []
        for subject_i in tqdm(subjects, position=0, leave=False):
            for subject_j in tqdm(subjects, position=1, leave=False):
                if subject_i == subject_j:
                    continue
                df_bold_i = df_bold.query("subject == @subject_i").copy()
                df_bold_j = df_bold.query("subject == @subject_j").copy()
                df_bold_rep = df_bold_i.merge(
                    df_bold_j, on=["bold_id", "repetition"], suffixes=("_i", "_j")
                )

                for stim_source in tqdm(stim_sources, position=2, leave=False):
                    df_bold_rep_source = df_bold_rep.copy()
                    if stim_source is not None:
                        df_bold_rep_source = df_bold_rep_source.query(
                            "stim_source_i == @stim_source"
                        )

                    df_results_pairs = measure_rsa(
                        df_bold_rep_source,
                        subject_i,
                        subject_j,
                        position=3,
                        join_hemispheres=join_hemispheres,
                        dataset="bold5000",
                        session_wise=session_wise,
                        q=quantile,
                    )
                    df_results_pairs["stim_source"] = stim_source
                    results.append(df_results_pairs)

        df_results = pd.concat(results, ignore_index=True)
        joined_suffix = "_joined" if join_hemispheres else "_separate"
        session_wise_suffix = "_session_wise" if session_wise else "_all"
        df_results.to_parquet(
            folder
            / f"bold5000_rsa_intersubject_diagonal{joined_suffix}{session_wise_suffix}.parquet",
            index=False,
        )


def compute_things_rsa(session_wise=True, quantile=0.003, folder=None):
    df = get_things_resource("stimulus")
    df = df[
        ["things_id", "subject_id", "session", "run", "subject_index", "repetition"]
    ]

    subjects = df["subject_id"].unique()

    for join_hemispheres in [False, True]:
        results = []
        for subject_i in tqdm(subjects, position=0, leave=False):
            for subject_j in tqdm(subjects, position=1, leave=False):
                if subject_i == subject_j:
                    continue
                df_things_i = df.query("subject_id == @subject_i").copy()
                df_things_j = df.query("subject_id == @subject_j").copy()
                df_things_rep = df_things_i.merge(
                    df_things_j, on=["things_id", "repetition"], suffixes=("_i", "_j")
                )
                df_things_res = measure_rsa(
                    df_things_rep,
                    subject_i,
                    subject_j,
                    dataset="things",
                    position=2,
                    q=quantile,
                    session_wise=session_wise,
                    join_hemispheres=join_hemispheres,
                )
                results.append(df_things_res)
        df_things_results = pd.concat(results, ignore_index=True)
        joined_suffix = "_joined" if join_hemispheres else "_separate"
        session_wise_suffix = "_session_wise" if session_wise else "_all"
        df_things_results.to_parquet(
            folder
            / f"things_rsa_intersubject_diagonal{joined_suffix}{session_wise_suffix}.parquet",
            index=False,
        )


if __name__ == "__main__":
    folder = Path("rsa_other_dataset")
    folder.mkdir(parents=True, exist_ok=True)
    for session_wise in [False]:
        compute_bold5000_rsa(
            session_wise=session_wise,
            quantile=QUANTILE,
            folder=folder,
        )
        compute_things_rsa(
            session_wise=session_wise,
            quantile=QUANTILE,
            folder=folder,
        )

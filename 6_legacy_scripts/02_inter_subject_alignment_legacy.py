"""
inter_subject_alignment_legacy.py

Legacy script to compute cross-subject alignment using only shared NSD stimuli,
matched by image identity and repetition (maintaining the same task structure).

Steps:
1. Extracts and saves common beta responses per subject as NIfTI volumes.
2. Joins batches into a full subject-wise beta file.
3. Computes RSA alignment between all subject pairs across ROIs.

Supports full ROIxROI pairwise comparison via --pairwise flag.
Results are saved as CSV in the specified destination folder.
"""

import argparse
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from convergence.nsd import get_index, load_betas, load_mask
import nibabel as nib
from convergence.metrics import measure

import torch
from dmf.alerts import alert, send_message


@alert(input=["subject"])
def generate_subject_common_betas(
    subject: int, dest_folder: Path, df_common: pd.DataFrame
) -> tuple:

    df_subject = df_common.query("subject == @subject").reset_index(drop=True).copy()
    sessions = list(df_subject.session.unique())

    batch_size = 10  # Number of sessions per NIfTI file
    all_betas_filtered = None
    batch_count = 0

    for i, session in enumerate(
        tqdm(sessions, desc=f"Processing subject {subject}", leave=False, position=1)
    ):
        df_session = df_subject.query(f"session == {session}")
        session_common_indexes = df_session.session_index.tolist()

        # Load betas for the session, now returns NIfTI object
        betas_nii = load_betas(subject=subject, session=session, return_nii=True)

        # Use memory-mapped data access with `.dataobj`
        data = betas_nii.get_fdata()

        if all_betas_filtered is None:
            all_betas_filtered = data[..., session_common_indexes]
        else:
            all_betas_filtered = np.concatenate(
                [all_betas_filtered, data[..., session_common_indexes]], axis=-1
            )

        del data
        gc.collect()

        # Every `batch_size` sessions, concatenate and save the file
        if (i + 1) % batch_size == 0 or i == len(sessions) - 1:
            # Concatenate the filtered betas for the batch

            # Create a new NIfTI image for the concatenated betas
            concatenated_nii = nib.Nifti1Image(
                all_betas_filtered, affine=betas_nii.affine
            )

            # Save the concatenated NIfTI image as a compressed `.nii.gz`
            file_name = (
                dest_folder
                / f"subject_{subject}_common_betas_batch_{batch_count}.nii.gz"
            )
            nib.save(concatenated_nii, file_name)
            del concatenated_nii, all_betas_filtered
            all_betas_filtered = None

            gc.collect()

            # Increment batch count
            batch_count += 1

    # Save the common beta indexes
    df_subject.to_csv(
        dest_folder / f"subject_{subject}_common_beta_indexes.csv", index=False
    )

    return True


def generate_common_betas(dest_folder: Path) -> None:
    df = get_index("stimulus")
    df_common = df.query("shared == True and flagged == False and exists == True")

    folder = Path(dest_folder)
    folder.mkdir(exist_ok=True)

    for subject in trange(1, 9, desc="Subjects", position=0):
        # Check if file already exists, if so, skip
        if (folder / f"subject_{subject}_common_betas.nii.gz").exists():
            continue

        # Call the function to process and save the common betas
        generate_subject_common_betas(subject, folder, df_common)


@alert(input=["subject"])
def join_subject_sessions(subject, folder):
    sessions = folder.glob(f"subject_{subject}_common_betas_batch_*.nii.gz")
    df = pd.read_csv(folder / f"subject_{subject}_common_beta_indexes.csv")

    # Open sessions
    sessions = [str(session) for session in sessions]
    send_message(f"Sessions: {sessions}")
    # Sort sessions by name
    sessions.sort()
    print(sessions)
    session0 = nib.load(sessions[0])
    data0 = session0.get_fdata()

    x, y, z, _ = data0.shape
    t = len(df)  # T is total length of betas
    dtype = data0.dtype
    del data0, session0
    gc.collect()
    data = np.zeros((x, y, z, t), dtype=dtype)

    counter = 0
    # Fill all concatenating last dimension
    for i, session in enumerate(sessions):
        session = nib.load(session)
        session_data = session.get_fdata()
        t_i = session_data.shape[-1]
        data[:, :, :, counter : counter + t_i] = session_data
        counter += t_i
        del session, session_data
        gc.collect()
        print(f"{t_i}/{t}")

    session0 = nib.load(sessions[0])
    send_message("Saving common betas")
    # Create nifti file
    nii = nib.Nifti1Image(data, session0.affine, session0.header)

    # Save nifti file as subject_{subject}_common_betas.nii.gz
    nib.save(nii, folder / f"subject_{subject}_common_betas.nii.gz")

    # Delete session files
    for session in sessions:
        Path(session).unlink()


def join_sessions(folder):
    for subject in range(1, 9):
        if (folder / f"subject_{subject}_common_betas.nii.gz").exists():
            send_message(f"Subject {subject} already exists")
            continue

        join_subject_sessions(subject, folder)


def extract_rois(betas, mask) -> dict[int, np.ndarray]:
    rois = {}
    for i in np.unique(mask):
        i = int(i)
        if i <= 0:
            continue
        v = betas[mask == i]
        a, b = np.quantile(v.ravel(), [0.01, 0.99])
        v = np.clip(v, a, b)
        rois[i] = v.T

    return rois


def order_betas(betas_rois, order):
    ordered_betas = {}
    for key, value in betas_rois.items():
        v = torch.tensor(value[order], device="cuda")
        ordered_betas[key] = v
    return ordered_betas


def load_masked_betas_indexes(
    subject, roi="HCP_MMP1", folder="common_betas", reduce_columns=True
):
    folder = Path(folder)
    mask = load_mask(subject=subject, roi=roi)
    betas = nib.load(folder / f"subject_{subject}_common_betas.nii.gz").get_fdata()
    betas_rois = extract_rois(betas, mask)

    indexes = pd.read_csv(
        folder / f"subject_{subject}_common_beta_indexes.csv"
    ).reset_index()
    if reduce_columns:
        indexes = indexes[["index", "nsd_id", "repetition"]]

    del betas
    gc.collect()
    return betas_rois, indexes


def free_betas(betas):
    for key, value in betas.items():
        value.cpu()
        del value
    del betas
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


@alert(input=["i", "j", "pairwise"])
def evaluate_main_alignment(
    betas_i, betas_j, indexes_i, indexes_j, i, j, pairwise=False, metric="rsa"
):
    merged = pd.merge(indexes_i, indexes_j, on=["nsd_id", "repetition"])
    betas_i_ordered = order_betas(betas_i, merged["index_i"].values)
    betas_j_ordered = order_betas(betas_j, merged["index_j"].values)

    cross_subject_alignment = []
    for roi_i in tqdm(
        list(betas_i_ordered.keys()), position=2, desc="ROIi", leave=False
    ):
        for roi_j in tqdm(
            list(betas_j_ordered.keys()), position=3, desc="ROIj", leave=False
        ):
            if pairwise or (roi_i == roi_j):
                score = measure(metric, betas_i_ordered[roi_i], betas_j_ordered[roi_j])
                cross_subject_alignment.append(
                    {
                        "subject_i": i,
                        "subject_j": j,
                        "roi_i": roi_i,
                        "roi_j": roi_j,
                        "score": score,
                        "metric": metric,
                    }
                )

    free_betas(betas_j_ordered)
    free_betas(betas_i_ordered)

    return cross_subject_alignment


@alert(output=True, input=["pairwise", "roi"])
def compute_main_alignment(folder, pairwise=False, roi="HCP_MMP1"):
    alignment = []
    try:

        for i in trange(1, 9, desc="Subject", leave=False, position=0):
            betas_i, indexes_i = load_masked_betas_indexes(i, folder=folder, roi=roi)
            indexes_i = indexes_i.rename(columns={"index": "index_i"})

            for j in trange(1, 9, desc="Subjectj", leave=False, position=1):
                if i < j:
                    continue

                betas_j, indexes_j = load_masked_betas_indexes(j, folder=folder, roi=roi)
                indexes_j = indexes_j.rename(columns={"index": "index_j"})

                cross_subject_alignment = evaluate_main_alignment(
                    betas_i,
                    betas_j,
                    i=i,
                    j=j,
                    indexes_i=indexes_i,
                    indexes_j=indexes_j,
                    pairwise=pairwise,
                )
                alignment.extend(cross_subject_alignment)

                del betas_j
                gc.collect()
            del betas_i
            gc.collect()

            # Save partial results
            df = pd.DataFrame(alignment)
            df.to_csv(folder / f"alignment_partial_results_{i}_{roi}.csv", index=False)

    except (Exception, KeyboardInterrupt) as e:
        send_message(f"Error: {e}")
        df = pd.DataFrame(alignment)
        df.to_csv(folder / "alignment_stopped.csv", index=False)
        raise e

    name = "pairwise_alignment" if pairwise else "main_alignment"
    df = pd.DataFrame(alignment)
    filename = folder / f"{name}_{roi}.csv"
    df.to_csv(filename, index=False)
    return filename


def main():
    parser = argparse.ArgumentParser(description="Generate common betas for subjects")
    parser.add_argument(
        "--dest",
        type=str,
        required=False,
        help="Destination folder for saving common betas",
        default="common_betas",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Compute pairwise alignment",
        default=False,
    )

    parser.add_argument(
        "--roi", type=str, required=False, help="ROI to use", default="HCP_MMP1"
    )

    args = parser.parse_args()
    pairwise = args.pairwise
    roi = args.roi

    dest = Path(args.dest)
    dest.mkdir(exist_ok=True)

    # Call the function to generate common betas
    # generate_common_betas(dest)
    # join_sessions(dest)
    compute_main_alignment(dest, pairwise=pairwise, roi=roi)


if __name__ == "__main__":
    main()

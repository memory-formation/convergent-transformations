"""
4_nsd_organize_betas.py

Extract and reorganize NSD single-trial betas into per-ROI NumPy arrays.

1. For each subject (1-8) and each session:
   a. Load the full-brain beta volumes via `convergence.nsd.load_betas`.
   b. Load the HCP-MMP1 ROI mask via `convergence.nsd.load_mask`.
   c. For each ROI, extract voxel time courses across trials and save as:
        sub-<subject>_ses-<session>_roi-<roi_id>.npy
      Shape: (n_trials, n_voxels_in_ROI).

2. After all sessions are processed, concatenate session files for each subject+ROI into:
        sub-<subject>_roi-<roi_id>.npy
   and delete the per-session files.

3. Finally, create per-subject subfolders (`sub01/`, ..., `sub08/`) and move all ROI `.npy` files 
into the corresponding directory.

Usage:
    python nsd_betas_to_roi_npy.py /path/to/output_folder

Requirements:
    - NSD betas and HCP ROI files accessible via `convergence.nsd` utilities.
    - `dmf.alerts` for progress alerts (optional; falls back gracefully).

"""
from pathlib import Path

import numpy as np
from tqdm import trange, tqdm

from dmf.alerts import alert, send_alert
from convergence.nsd import load_betas, load_mask, get_resource, get_index


def save_session(subject, session, mask, betas, folder, rois):
    for roi in tqdm(rois, leave=False, position=2, desc="ROI"):
        roi_values = betas[mask == roi].T  # shape: (trial, voxel)
        assert roi_values.shape[0] == betas.shape[-1]

        filename = folder / f"sub-{subject:02}_ses-{session:02}_roi-{roi:03}.npy"
        np.save(filename, roi_values)


def join_sessions(folder, subject, rois):
    """Get all files from a subject and join them in session order by roi"""

    for roi in tqdm(rois, leave=False, position=2, desc="Join ROI"):
        filename = folder / f"sub-{subject:02}_roi-{roi:03}.npy"
        files = sorted(folder.glob(f"sub-{subject:02}_ses-*_roi-{roi:03}.npy"))
        if filename.exists():
            continue

        data = [np.load(file) for file in files]
        data = np.concatenate(data, axis=0)

        np.save(filename, data)

        # remove session files
        for file in files:
            file.unlink()


def create_subfolders(folder):
    "Create a subfolder for each subject and move the files"
    # Glob with f"sub-{subject:02}_roi-{roi:03}.npy" pattern
    for participant in range(1, 9):
        subfolder = folder / f"sub{participant:02}"
        subfolder.mkdir(exist_ok=True)
        files = folder.glob(f"sub-{participant:02}_roi-*.npy")
        for file in files:
            file.rename(subfolder / file.name)


@alert
def main(folder):
    rois = get_resource("hcp").roi.tolist()
    df = (
        get_index("stimulus")
        .query("exists == True")[["subject", "session"]]
        .drop_duplicates()
    )

    for subject in trange(1, 9, leave=False, position=0, desc="Subject"):
        send_alert(f"Processing subject {subject}")
        sessions = df.query("subject == @subject").session.tolist()
        mask = load_mask(subject=subject, roi="both.HCP_MMP1")
        for session in tqdm(sessions, leave=False, position=1, desc="Session"):
            betas = load_betas(subject=subject, session=session).astype("int16")
            save_session(subject, session, mask, betas, folder, rois)

        # Save joined sessions
        join_sessions(folder, subject, rois)


if __name__ == "__main__":

    folder = "/mnt/tecla/Datasets/nsd/betas"
    folder = Path(folder)
    folder.mkdir(exist_ok=True)

    main(folder=folder)
    create_subfolders(folder)

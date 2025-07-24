"""
1_nsd_check_dataset.py

Validate the presence of required Natural Scenes Dataset (NSD) files.

This script verifies that all essential NSD files are available in the specified dataset directory,
including:
  - Stimulus files (`nsd_stimuli.hdf5`, `nsd_stim_info_merged.csv`)
  - Per-subject ROI masks (HCP_MMP1, left & right hemisphere)
  - Whole-brain mask
  - Session beta maps for each subject

Usage:
  python nsd_check_dataset.py --folder /path/to/NSD_DATASET

Alternatively, set the NSD_DATASET environment variable:
  export NSD_DATASET=/path/to/NSD_DATASET
  python nsd_check_dataset.py

If any file is missing, the script will report errors and list the missing paths.
"""
from pathlib import Path
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Global variables
RESOLUTION = "func1mm"
BETAS_VERSION = "betas_fithrf_GLMdenoise_RR"
ROI_SUBPATH = "roi"
ROIS = ["HCP_MMP1.nii.gz", "lh.HCP_MMP1.nii.gz", "rh.HCP_MMP1.nii.gz"]
BRAIN_MASK = "brainmask.nii.gz"
STIMULUS_FILE = "nsd_stimuli.hdf5"
STIMULUS_INFO_FILE = "nsd_stim_info_merged.csv"

# ANSI escape codes for colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Define the subjects and sessions
subjects = {
    "subj01": 40,
    "subj02": 40,
    "subj03": 32,
    "subj04": 30,
    "subj05": 40,
    "subj06": 32,
    "subj07": 40,
    "subj08": 30,
}

not_found = []


def exists(file_path, description):

    if file_path.exists():
        logging.info(f"{GREEN}[OK]{RESET} {description} found.")
    else:
        logging.info(f"{RED}[ERROR]{RESET} {description} not found!")
        not_found.append(file_path)


# Parse command-line arguments to get the NSD_DATASET path
def parse_args():
    parser = argparse.ArgumentParser(description="Check for missing NSD dataset files.")
    parser.add_argument(
        "--folder", type=str, help="Path to the NSD dataset folder", required=False
    )

    args = parser.parse_args()
    nsd_dataset = args.folder
    if not nsd_dataset:
        nsd_dataset = os.getenv("NSD_DATASET")

    if not nsd_dataset:
        parser.error(
            "NSD dataset path not provided. "
            "Pass it as an argument or set the NSD_DATASET environment variable."
        )
        exit(1)

    return Path(nsd_dataset).expanduser().resolve()


def main():
    # Get the NSD dataset path from command-line argument
    base_dir = parse_args()

    # Check stimulus files
    stimulus_files = [STIMULUS_FILE, STIMULUS_INFO_FILE]
    for stim_file in stimulus_files:
        stim_path = base_dir / stim_file
        exists(stim_path, f"Stimulus file '{stim_file}'")

    # Loop through each subject and their sessions
    for subject, session_count in subjects.items():
        subject_dir = base_dir / subject / RESOLUTION

        # Check for ROI files (HCP_MMP1 and others) in func1mm
        for roi_file in ROIS:
            roi_path = base_dir / subject / RESOLUTION / ROI_SUBPATH / roi_file
            exists(roi_path, f"{subject} - Mask '{roi_file}'")

        # Check for the whole brain mask file
        brainmask_path = base_dir / subject / RESOLUTION / "brainmask.nii.gz"
        exists(brainmask_path, f"{subject} - Whole brain mask")

        # Loop through each session for the subject
        for session_num in range(1, session_count + 1):
            session_file = f"betas_session{session_num:02d}.nii.gz"
            session_path = subject_dir / BETAS_VERSION / session_file
            exists(session_path, f"{subject} - Session {session_num} Beta")

    if not_found:
        logging.info(f"Missing {RED}{len(not_found)}{RESET} files:")
        for file in not_found:
            logging.info(f" - {RED}{file.resolve()}{RESET}")
    else:
        logging.info(f"{GREEN}All files found!{RESET}")


if __name__ == "__main__":
    main()

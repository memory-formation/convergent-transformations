"""
2_nsd_create_indexes.py

Organize the Natural Scenes Dataset (NSD) for analysis by:
  1. Unrolling the HDF5 stimulus file into individual JPEG images, grouped into folders of 1000.
  2. Generating an image index CSV (`nsd.csv`) listing image IDs, filenames, and relative paths.
  3. Building a trial-level stimuli index (`nsd_stimuli.csv`) that maps each subject's trial order,
     session, and repetition to NSD image IDs and metadata.
  4. Creating an ROI mask index (`nsd_masks.csv`) listing available ROI mask files per subject
     and hemisphere.

Usage:
  python nsd_prepare_dataset.py --folder /path/to/NSD_DATASET

Or set the NSD_DATASET environment variable:
  export NSD_DATASET=/path/to/NSD_DATASET
  python nsd_prepare_dataset.py

The script expects the following at the dataset root:
  - `nsd_stimuli.hdf5` containing the image brick.
  - `nsd_stim_info_merged.csv` with trial metadata.
  - A subfolder `images/` will be created for JPEG outputs.
  - Subject folders `subj01/`…`subj08/` with `func1mm/roi/` masks.

Outputs:
  - `nsd.csv`: Image index.
  - `nsd_stimuli.csv`: Trial-level stimuli index for all subjects.
  - `nsd_masks.csv`: ROI mask file index.
"""
from pathlib import Path
import os
import h5py
from tqdm import trange
from PIL import Image
import argparse
import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

STIMULUS_FILE = "nsd_stimuli.hdf5"
IMAGE_GROUPS = 1000
IMAGES_INDEX = "nsd.csv"

STIMULI_INFO = "nsd_stim_info_merged.csv"
NUM_IMAGES_SESSION = 750
STIMULI_OUTPUT = "nsd_stimuli.csv"

MASK_INDEX = "nsd_masks.csv"


def parse_args():
    args = argparse.ArgumentParser(description="Structure NSD dataset.")
    args.add_argument(
        "--folder", type=str, help="Path to the NSD dataset folder", required=False
    )
    args = args.parse_args()
    nsd_dataset = args.folder
    if not nsd_dataset:
        nsd_dataset = os.getenv("NSD_DATASET")

    if not nsd_dataset:
        args.error(
            "NSD dataset path not provided. "
            "Pass it as an argument or set the NSD_DATASET environment variable."
        )
        exit(1)

    return Path(nsd_dataset).expanduser().resolve()


def unroll_dataset(base_dir):

    images_file = base_dir / STIMULUS_FILE
    images_folder = base_dir / "images"
    images_folder.mkdir(exist_ok=True)

    if not images_file.exists():
        logging.error(f"{images_file} not found!")
        exit(1)

    image_list = []

    with h5py.File(images_file, "r") as file:
        matrix = file["imgBrick"]
        num_images = matrix.shape[0]
        assert num_images == 73_000

        for i in trange(num_images, desc="Unrolling images", leave=False):
            # Compute block number and image path
            block_num = i // IMAGE_GROUPS
            block_folder = (
                images_folder
                / f"nsd_{block_num * IMAGE_GROUPS:05d}_{(block_num + 1) * IMAGE_GROUPS - 1:05d}"
            )
            block_folder.mkdir(exist_ok=True)

            image = matrix[i]
            image_path = block_folder / f"nsd_{i:05d}.jpg"
            if not image_path.exists():
                Image.fromarray(image).save(image_path)

            image_list.append(
                {
                    "nsd_id": i,
                    "name": image_path.stem,
                    "path": image_path.relative_to(base_dir),
                }
            )

    # Save the image list to a CSV file
    df = pd.DataFrame(image_list)
    images_index_path = base_dir / IMAGES_INDEX
    df.to_csv(images_index_path, index=False)
    logging.info(f"Generated image index: {images_index_path}")


def get_subject(
    subject: int, df: pd.DataFrame, num_images_session: int
) -> pd.DataFrame:
    """Process the data for a single subject."""
    assert subject in range(1, 9), "subject must be in range 1-8"
    col_identifier = f"subject{subject}"
    df_s = df[df[col_identifier] == 1].copy()

    # Keep columns that start with 'subject{subject}' or do not start with 'subject'
    keep = [
        col
        for col in df_s.columns
        if (col.startswith("subject") and col.startswith(col_identifier))
        or not col.startswith("subject")
    ]
    df_s = df_s[keep].drop(columns=col_identifier)

    # Drop columns that start with 'Unnamed'
    df_s = df_s.loc[:, ~df_s.columns.str.contains("^Unnamed")]

    # Rename columns for clarity
    rename_columns = {
        "cocoId": "coco_id",
        "cocoSplit": "coco_split",
        "cropBox": "crop_box",
        "BOLD5000": "bold5000",
        "nsdId": "nsd_id",
        "shared1000": "shared",
        f"{col_identifier}_rep0": "0",
        f"{col_identifier}_rep1": "1",
        f"{col_identifier}_rep2": "2",
    }
    df_s = df_s.rename(columns=rename_columns).drop(columns=["loss"])

    # Move repetition columns (0, 1, 2) to rows
    df_s = df_s.melt(
        id_vars=[col for col in df_s.columns if not col.isdigit()],
        var_name="repetition",
        value_name="subject_index",
    )

    # Add session information and sort
    df_s["subject"] = subject
    df_s["session"] = np.ceil(df_s["subject_index"] / num_images_session).astype(int)
    df_s = df_s.sort_values(by=["subject_index", "nsd_id", "repetition"])
    df_s["session_index"] = df_s.groupby("session").cumcount() + 1

    # Organize final columns
    cols = [
        "subject",
        "subject_index",
        "session",
        "session_index",
        "nsd_id",
        "repetition",
        "flagged",
        "shared",
        "bold5000",
        "coco_id",
        "coco_split",
        "crop_box",
    ]
    df_s["subject_index"] -= 1
    df_s["session_index"] -= 1
    df_s = (
        df_s[cols]
        .sort_values(by=["subject_index", "nsd_id", "repetition"])
        .reset_index(drop=True)
    )

    assert df_s.subject_index.is_unique, "subject_index is not unique"
    return df_s


def get_session_file(
    betas_folder,
    subject,
    session,
    subj_template="subj{subject:02d}",
    resolution="func1mm",
    preprocessing="betas_fithrf_GLMdenoise_RR",
    session_template="betas_session{session:02d}",
    extension=".nii.gz",
):
    assert subject in range(1, 9), "subject must be in range 1-8"

    filename = (
        Path(betas_folder)
        / subj_template.format(subject=subject)
        / resolution
        / preprocessing
        / (session_template.format(session=session) + extension)
    )
    # assert filename.exists(), f"File {filename} does not exist"
    return filename


def get_session_files(
    betas_folder,
    subj_template="subj{subject:02d}",
    resolution="func1mm",
    preprocessing="betas_fithrf_GLMdenoise_RR",
    session_template="betas_session{session:02d}",
    extension=".nii.gz",
):
    filenames = []
    for subject in range(1, 9):
        for session in range(1, 41):
            filename = get_session_file(
                betas_folder,
                subject,
                session,
                subj_template=subj_template,
                resolution=resolution,
                preprocessing=preprocessing,
                session_template=session_template,
                extension=extension,
            )
            filenames.append(
                {
                    "subject": subject,
                    "session": session,
                    "filename": filename.relative_to(betas_folder),
                    "exists": filename.exists(),
                }
            )

    df = pd.DataFrame(filenames)
    return df


def generate_stimuli_index(base_dir):
    """Process stimuli data for all subjects and save to the output file."""
    df_nsd = pd.read_csv(base_dir / STIMULI_INFO)
    subjects = []
    for subject in range(1, 9):
        df_s = get_subject(subject, df_nsd, NUM_IMAGES_SESSION)
        subjects.append(df_s)

    # Concatenate all subjects' data and save to CSV
    df = pd.concat(subjects).reset_index(drop=True)

    df_files = get_session_files(base_dir)
    df = df.merge(df_files, on=["subject", "session"], how="left")

    stimuli_index = base_dir / STIMULI_OUTPUT
    df.to_csv(stimuli_index, index=False)
    logging.info(f"Generated stimuli index: {stimuli_index}")


def get_masks_files(
    mask_folder,
    subj_template="subj{subject:02d}",
    resolution="func1mm",
    roi_folder="roi",
):
    mask_folder = Path(mask_folder)
    rois = []
    for subj in range(1, 9):
        subject_masks = (
            mask_folder / subj_template.format(subject=subj) / resolution / roi_folder
        )
        sub_rois = list(subject_masks.glob("*.nii.gz"))
        for roi in sub_rois:
            rois.append(
                {
                    "subject": subj,
                    "roi": roi.stem.replace(".nii", ""),
                    "mask_path": roi.relative_to(mask_folder),
                }
            )

    df = pd.DataFrame(rois)
    # Hemisphere: right if start with rh., left if start with lh. else both
    df["hemisphere"] = df["roi"].apply(
        lambda x: "r" if x.startswith("rh.") else "l" if x.startswith("lh.") else "b"
    )
    # Sort columns. mask at the end
    df = df[["subject", "roi", "hemisphere", "mask_path"]]
    mask_file = mask_folder / MASK_INDEX
    df.to_csv(mask_file, index=False)
    logging.info(f"Generated masks index: {mask_file}")


def main():

    base_dir = parse_args()

    # Prepare the images
    unroll_dataset(base_dir)

    # Prepare the stimuli index
    generate_stimuli_index(base_dir)

    # Prepare the masks index
    get_masks_files(base_dir)


if __name__ == "__main__":
    main()

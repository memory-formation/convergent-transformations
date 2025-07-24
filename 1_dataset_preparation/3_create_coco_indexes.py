"""
3_create_coco_indexes.py

Download COCO annotations and generate NSD‐compatible index files:

1. Downloads and extracts COCO 2017 annotations (captions, instances, keypoints) into a specified folder.
2. Builds a merged COCO->NSD caption index (`coco_captions.csv`), mapping COCO image IDs to NSD IDs.
3. Builds a COCO object annotations index (`coco_objects_annotations.csv`), listing category, area, and bounding boxes per NSD image.
4. Builds a COCO person keypoints index (`coco_person_annotations.csv`), extracting and formatting keypoint coordinates for each NSD image.

Default paths assume:
  - NSD CSV at ~/Datasets/nsd/nsd_stim_info_merged.csv (change as needed).
  - Download folder at ./data (change as needed).
  - COCO JSONs under ./data/annotations/ (change as needed).

All output CSVs will be overwritten if they already exist.
"""
import argparse
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np

from tqdm import tqdm
import pandas as pd
import numpy as np

URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_coco_annotations(url, dest):
    # Set up destination path and filename
    dest_path = Path(dest)
    filename = dest_path / Path(url).name

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Check if file already exists before downloading
    if not filename.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

        # Unzip the downloaded file into the destination folder
        print("Unzipping the file...")
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(dest_path)

        # Remove the zip file after extraction
        
        print("Download and extraction complete. Zip file removed.")
    else:
        print(f"{filename} already exists, skipping download.")


def generate_caption_index(
    nsd_file, val_captions_file, train_captions_file, caption_file
):
    # Load NSD stimulus information
    df = pd.read_csv(nsd_file)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Load COCO captions for validation and training splits
    with open(val_captions_file) as f:
        val_data = json.load(f)

    with open(train_captions_file) as f:
        train_data = json.load(f)

    # Convert JSON annotations to DataFrames
    val_annot = pd.DataFrame(val_data["annotations"])
    train_annot = pd.DataFrame(train_data["annotations"])

    # Process captions by replacing semicolons and grouping by image_id
    val_annot["caption"] = val_annot["caption"].str.replace(";", ",")
    val_annot = (
        val_annot.groupby("image_id")["caption"]
        .apply(lambda x: ";".join(x))
        .reset_index()
    )
    val_annot["cocoSplit"] = "val2017"

    train_annot["caption"] = train_annot["caption"].str.replace(";", ",")
    train_annot = (
        train_annot.groupby("image_id")["caption"]
        .apply(lambda x: ";".join(x))
        .reset_index()
    )
    train_annot["cocoSplit"] = "train2017"

    # Concatenate validation and training captions
    captions = pd.concat([val_annot, train_annot])
    captions = captions.rename(columns={"image_id": "coco_id"})

    # Merge with NSD DataFrame based on COCO ID, rename and reorder columns
    captions = captions.merge(
        df[["cocoId", "nsdId"]], left_on="coco_id", right_on="cocoId", how="inner"
    )
    captions = captions.rename(columns={"nsdId": "nsd_id", "caption": "captions"}).drop(
        columns="cocoId"
    )
    captions = captions[["nsd_id", "coco_id", "cocoSplit", "captions"]]
    captions.to_csv(caption_file, index=False)

    return captions


def generate_object_index(nsd_file, val_objects_file, train_objects_file, filename):
    # Load NSD stimulus information
    df = pd.read_csv(nsd_file).drop(columns=["Unnamed: 0"], errors="ignore")
    translation = df[["cocoId", "nsdId"]]

    # Load validation objects and process annotations
    with open(val_objects_file, "r") as f:
        val_data = json.load(f)
    val_annot = pd.DataFrame(val_data["annotations"])
    val_categories = pd.DataFrame(val_data["categories"])

    val_objects = val_annot[val_annot.image_id.isin(translation.cocoId)]
    val_objects = val_objects[["image_id", "category_id", "area", "bbox"]]
    val_objects = val_objects.merge(
        val_categories, left_on="category_id", right_on="id"
    ).drop(columns=["id"])
    val_objects["coco_split"] = "val2017"
    val_objects = val_objects.rename(columns={"name": "category"})

    # Load training objects and process annotations
    with open(train_objects_file, "r") as f:
        train_data = json.load(f)
    train_annot = pd.DataFrame(train_data["annotations"])
    train_categories = pd.DataFrame(train_data["categories"])

    train_objects = train_annot[train_annot.image_id.isin(translation.cocoId)]
    train_objects = train_objects[["image_id", "category_id", "area", "bbox"]]
    train_objects = train_objects.merge(
        train_categories, left_on="category_id", right_on="id"
    ).drop(columns=["id"])
    train_objects["coco_split"] = "train2017"
    train_objects = train_objects.rename(columns={"name": "category"})

    # Concatenate validation and training objects
    objects_annotations = pd.concat([val_objects, train_objects])

    # Merge with NSD DataFrame
    objects_annotations = (
        objects_annotations.merge(translation, left_on="image_id", right_on="cocoId")
        .drop(columns=["cocoId"])
        .rename(columns={"nsdId": "nsd_id", "image_id": "coco_id"})
    )

    # Final formatting and saving
    objects_annotations = objects_annotations[
        [
            "nsd_id",
            "coco_id",
            "coco_split",
            "category",
            "supercategory",
            "category_id",
            "area",
            "bbox",
        ]
    ]
    objects_annotations["area"] = objects_annotations["area"].astype(int)
    objects_annotations["bbox"] = (
        objects_annotations["bbox"]
        .astype(str)
        .str.replace("[", "(")
        .str.replace("]", ")")
    )
    objects_annotations = objects_annotations.sort_values(by="nsd_id").reset_index(
        drop=True
    )

    # Save to CSV
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    objects_annotations.to_csv(output_path, index=False)


def generate_person_index(nsd_file, val_person_file, train_person_file, output_file):
    # Load NSD stimulus information
    df_nsd = pd.read_csv(nsd_file)
    df_translate = df_nsd[['cocoId', 'cocoSplit', "nsdId"]].drop_duplicates()
    assert df_translate.cocoId.is_unique

    # Process person keypoints for validation split
    with open(val_person_file, "r") as f:
        val_data = json.load(f)
    keypoint_names = val_data['categories'][0]['keypoints']
    df_val = pd.DataFrame(val_data['annotations'])
    df_val['keypoints'] = df_val['keypoints'].apply(lambda kp: process_keypoints(kp, keypoint_names))
    df_val = df_val.merge(df_translate, left_on="image_id", right_on="cocoId")
    df_val = df_val.rename(columns={"nsdId": "nsd_id", "cocoId": "coco_id", "cocoSplit": "coco_split"})
    df_val["person_image_id"] = df_val.groupby("nsd_id").cumcount()
    df_val = df_val[["nsd_id", "coco_id", "coco_split", "person_image_id", "num_keypoints", "keypoints", "area", "bbox", "segmentation"]]
    df_val = df_val.sort_values(["nsd_id", "area"], ascending=[True, False])

    # Process person keypoints for training split
    with open(train_person_file, "r") as f:
        train_data = json.load(f)
    df_train = pd.DataFrame(train_data['annotations']).query("num_keypoints > 0")
    df_train['keypoints'] = df_train['keypoints'].apply(lambda kp: process_keypoints(kp, keypoint_names))
    df_train = df_train.merge(df_translate, left_on="image_id", right_on="cocoId")
    df_train = df_train.rename(columns={"nsdId": "nsd_id", "cocoId": "coco_id", "cocoSplit": "coco_split"})
    df_train["person_image_id"] = df_train.groupby("nsd_id").cumcount()
    df_train = df_train[["nsd_id", "coco_id", "coco_split", "person_image_id", "num_keypoints", "keypoints", "area", "bbox", "segmentation"]]
    df_train = df_train.sort_values(["nsd_id", "area"], ascending=[True, False])

    # Concatenate validation and training data
    df_person = pd.concat([df_val, df_train]).sort_values(["nsd_id", "person_image_id"])
    df_person["area"] = df_person["area"].astype(int)
    df_person = df_person.reset_index(drop=True)

    # Expand keypoints into individual columns
    for name in keypoint_names:
        df_person[name] = df_person["keypoints"].apply(lambda x: x.get(name, None))
    df_person = df_person.drop(columns=["keypoints"])

    # Final reordering and saving
    df_person = df_person[[
        'nsd_id', 'coco_id', 'coco_split', 'person_image_id', 'num_keypoints',
        'area', 'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
        'right_knee', 'left_ankle', 'right_ankle', 'bbox', 'segmentation'
    ]]
    df_person.to_csv(output_file, index=False)


def process_keypoints(keypoints, keypoint_names):
    """
    Process keypoints by filtering out invalid points and extracting 2D coordinates.
    
    Parameters:
    - keypoints: List of keypoints [x, y, visibility]
    - keypoint_names: List of keypoint names
    
    Returns:
    - Dictionary with keypoint names as keys and (x, y) tuples as values
    """
    keypoints = np.array(keypoints).reshape(-1, 3)
    keypoint_labels = {}
    for i, keypoint in enumerate(keypoints):
        if keypoint[2] >= 1:
            keypoint_labels[keypoint_names[i]] = keypoint[:2]
    return keypoint_labels


def main():
    parser = argparse.ArgumentParser(
        description="Process COCO dataset annotations and NSD mapping."
    )

    # Default values
    default_url = URL
    default_nsd_file = Path("~/Datasets/nsd/nsd_stim_info_merged.csv").expanduser()
    default_dest = Path("./data")
    default_val_captions_file = default_dest / "annotations/captions_val2017.json"
    default_train_captions_file = default_dest / "annotations/captions_train2017.json"
    default_caption_file = default_dest / "coco_captions.csv"
    default_val_objects_file = default_dest / "annotations/instances_val2017.json"
    default_train_objects_file = default_dest / "annotations/instances_train2017.json"
    default_object_file = default_dest / "coco_objects_annotations.csv"
    default_val_person_file = default_dest / "annotations/person_keypoints_val2017.json"
    default_train_person_file = default_dest / "annotations/person_keypoints_train2017.json"
    default_person_file = default_dest / "coco_person_annotations.csv"

    # Argument definitions with default values
    parser.add_argument("--url", type=str, default=default_url, help="URL to download COCO annotations")
    parser.add_argument("--nsd_file", type=Path, default=default_nsd_file, help="Path to the NSD CSV file")
    parser.add_argument("--dest", type=Path, default=default_dest, help="Destination folder for COCO data")
    parser.add_argument("--val_captions_file", type=Path, default=default_val_captions_file, help="Path to COCO validation captions file")
    parser.add_argument("--train_captions_file", type=Path, default=default_train_captions_file, help="Path to COCO training captions file")
    parser.add_argument("--caption_file", type=Path, default=default_caption_file, help="Output file for caption data")
    parser.add_argument("--val_objects_file", type=Path, default=default_val_objects_file, help="Path to COCO validation objects file")
    parser.add_argument("--train_objects_file", type=Path, default=default_train_objects_file, help="Path to COCO training objects file")
    parser.add_argument("--object_file", type=Path, default=default_object_file, help="Output file for object annotations")
    parser.add_argument("--val_person_file", type=Path, default=default_val_person_file, help="Path to COCO validation person keypoints file")
    parser.add_argument("--train_person_file", type=Path, default=default_train_person_file, help="Path to COCO training person keypoints file")
    parser.add_argument("--person_file", type=Path, default=default_person_file, help="Output file for person annotations")

    args = parser.parse_args()

    # Download COCO annotations
    pbar = tqdm(total=4)
    pbar.set_description("Downloading COCO annotations")
    download_coco_annotations(args.url, args.dest)
    pbar.update(1)

    # Generate COCO caption indexes
    pbar.set_description("Generating COCO caption indexes")
    generate_caption_index(
        args.nsd_file,
        args.val_captions_file,
        args.train_captions_file,
        args.caption_file,
    )
    pbar.update(1)

    # Generate COCO object indexes
    pbar.set_description("Generating COCO object indexes")
    generate_object_index(
        args.nsd_file, args.val_objects_file, args.train_objects_file, args.object_file
    )
    pbar.update(1)

    # Generate COCO person indexes
    pbar.set_description("Generating COCO person indexes")
    generate_person_index(
        nsd_file=args.nsd_file,
        val_person_file=args.val_person_file,
        train_person_file=args.train_person_file,
        output_file=args.person_file
    )
    pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()

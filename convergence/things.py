from pathlib import Path
from typing import Optional, Union, Literal, Callable
import os
import pandas as pd
import nibabel as nib
import numpy as np
import datasets
from datasets import Dataset


__all__ = [
    "resolve_things_path",
    "get_resource",
    "load_mask",
    "get_betas_roi",
]

RESOURCE_FILENAMES = {
    "stimulus": "preprocessed/stimulus_index.parquet",
    "images": "preprocessed/images.csv",
    "categories": "preprocessed/categories.csv",
    "captions": "preprocessed/captions.csv",
    "hcp": "preprocessed/hcp.csv",
    "model-info": "model-info.csv",
}

MASK_ROIS = {"both.HCP_MMP1": "mask-glasser.nii.gz"}
BETAS_ROIS = {"both.HCP_MMP1": "betas_roi/betas_glasser_roi_{roi}.npy"}


THINGS_DATASET = os.getenv("THINGS_DATASET")

THINGS_CITATION = """Martin N Hebart, Oliver Contier, Lina Teichmann, Adam H Rockter, 
Charles Y Zheng, Alexis Kidder, Anna Corriveau, Maryam Vaziri-Pashkam, 
Chris I Baker (2023) THINGS-data, a multimodal collection of large-scale datasets for investigating 
object representations in human brain and behavior eLife 12:e82580"""


def resolve_things_path(base_dir: Optional[Path] = None) -> Path:
    """
    Resolves the THINGS dataset base directory.

    Args:
        base_dir (Optional[Path]): Custom path to the dataset folder. If None, attempts to use
        the `THINGS_DATASET` environment variable.

    Returns:
        Path: The resolved path to the NSD dataset.

    Raises:
        ValueError: If the path is not provided and `THINGS_DATASET` is not set.
    """
    base_dir = base_dir or THINGS_DATASET
    if not base_dir:
        raise ValueError(
            "THINGS dataset path not provided. "
            "Pass it as an argument or set the THINGS_DATASET environment variable."
        )
    return Path(base_dir).expanduser().resolve()


def get_resource(
    resource: Union[
        Literal[
            "stimulus",
            "hcp",
            "images",
            "categories",
            "captions",
            "model-info",
        ],
        str,
    ],
    base_dir: Optional[Path] = None,
    return_path: bool = False,
) -> Union[pd.DataFrame, Path]:
    """
    Loads a resource file for the THINGS dataset as a Pandas DataFrame.

    Args:
        resource (Literal["stimulus", "images", "categories", "captions", "model-info"]): The resource to load.
        base_dir (Optional[Path]): The base directory where the dataset is located. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame containing the resource data.

    Raises:
        AssertionError: If the resource file does not exist.
    """
    base_dir = resolve_things_path(base_dir)

    if resource not in RESOURCE_FILENAMES:
        raise ValueError(
            f"Resource {resource} not found in the NSD dataset. "
            f"Valid resources are: {list(RESOURCE_FILENAMES.keys())}"
        )

    filename = base_dir / RESOURCE_FILENAMES[resource]
    assert (
        filename.exists()
    ), f"{filename} not found! Did you complete the NSD dataset download and setup?"

    if return_path:
        return filename
    # If csv file, return as pandas dataframe
    if filename.suffix == ".csv":
        return pd.read_csv(filename)
    # If tsv file, return as pandas dataframe
    if filename.suffix == ".tsv":
        return pd.read_csv(filename, sep="\t")
    if filename.suffix == ".parquet":
        return pd.read_parquet(filename)

    from dmf.io import load

    return load(filename)


def load_mask(
    subject: int,
    roi: Literal["both.HCP_MMP1"] = "both.HCP_MMP1",
    base_dir: Optional[Path] = None,
    return_path: bool = False,
    return_nii: bool = False,
):
    base_dir = resolve_things_path(base_dir)

    if roi not in MASK_ROIS:
        raise ValueError(
            f"ROI {roi} not found in the THINGS dataset. "
            f"Valid ROIs are: {list(MASK_ROIS.keys())}"
        )

    assert subject in [1, 2, 3]
    mask_name = MASK_ROIS[roi]
    mask_path = base_dir / f"preprocessed/sub-0{subject}/" / mask_name
    assert mask_path.exists(), f"Mask file not found at {mask_path}"
    if return_path:
        return mask_path

    image = nib.load(mask_path)
    if return_nii:
        return image
    return image.get_fdata()


def get_betas_roi(
    subject: int,
    roi: Union[int, list[int]],
    atlas: Literal["both.HCP_MMP1"] = "both.HCP_MMP1",
    base_dir: Optional[Path] = None,
) -> np.ndarray:
    base_dir = resolve_things_path(base_dir)

    if atlas not in BETAS_ROIS:
        raise ValueError(
            f"ROI {atlas} not found in the THINGS dataset. "
            f"Valid ROIs are: {list(BETAS_ROIS.keys())}"
        )

    assert subject in [1, 2, 3]
    if isinstance(roi, int):
        roi = [roi]
    betas = []
    for r in roi:
        betas_name = BETAS_ROIS[atlas].format(roi=r)
        betas_path = base_dir / f"preprocessed/sub-0{subject}/" / betas_name
        assert betas_path.exists(), f"Betas file not found at {betas_path}"
        betas.append(np.load(betas_path))

    if len(betas) == 1:
        return betas[0]
    # Stack columns
    return np.hstack(betas)

def load_dataset(
    query: Optional[Union[Callable, str]] = None,
    cast_column: bool = True,
    df_index: Optional[pd.DataFrame] = None,
    base_dir: Optional[Union[Path, str]] = None,
) -> "Dataset":
    """
    Load a dataset from a directory, using a CSV index to map image paths.

    This function simplifies the loading of a dataset from a directory,
    assuming a CSV file that maps image paths. The dataset is returned as a
    Hugging Face `datasets.Dataset` object. The CSV is expected to contain a
    column named 'path' which holds the relative paths to the images.

    Args:
        query (Optional[Union[Callable, str]]): A filter to apply to the dataset.
            Can be a callable that takes a DataFrame and returns a filtered
            DataFrame, or a string representing a query that can be passed to
            `DataFrame.query()`. Defaults to None.
        cast_column (bool): Whether to cast the 'image' column as a `datasets.Image()`
            object (i.e., image type in Hugging Face datasets). Defaults to True.
        df_index (Optional[pd.DataFrame]): A preloaded DataFrame containing the index
            of images. If not provided, it will load the index from the NSD dataset
            ('images' index). Defaults to None.
        base_dir (Optional[Union[Path, str]]): Base directory where the dataset and
            images are located. If not provided, it will attempt to resolve the path
            using the `NSD_DATASET` environment variable or a default path. Defaults
            to None.

    Returns:
        datasets.Dataset: A Hugging Face `Dataset` object with the images loaded
        and paths adjusted according to the `base_dir`.

    Example:
        >>> from neuroplatonic.utils import load_dataset
        >>> dataset = load_dataset(query="subject == 1 and session == 1")
        >>> print(dataset)

    Notes:
        - The function expects a CSV file that contains a column named 'path' for
          image paths. The paths will be converted to absolute paths based on the
          `base_dir`.
        - If a query is provided, the dataset will be filtered accordingly. The
          query can be either a callable function or a string query.
        - If `cast_column` is True, the 'image' column will be cast to the
          `datasets.Image()` type, enabling easy image handling in the Hugging Face
          dataset.
    """

    base_dir = resolve_things_path(base_dir)
    df = df_index or get_resource("images")

    df["image"] = df["image_path"]
    df["image"] = df["image"].apply(lambda image_path: str(base_dir / image_path))

    if query is not None:
        if callable(query):
            df = query(df)
        else:
            df = df.query(query)

    # Create the dataset
    dataset = datasets.Dataset.from_pandas(df)

    if cast_column:
        dataset = dataset.cast_column("image", datasets.Image())

    # Add metadata to the dataset object
    dataset.info.citation = " ".join(THINGS_CITATION.split())
    dataset.info.dataset_name = "things"

    return dataset

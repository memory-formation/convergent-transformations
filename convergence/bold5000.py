from pathlib import Path
from typing import Optional, Union, Literal, Callable
import os
import pandas as pd
import nibabel as nib
import numpy as np
import datasets
from datasets import Dataset

RESOURCE_FILENAMES = {
    "stimulus": "preprocessed/stimulus_index.parquet",
    "images": "preprocessed/images.parquet",
    "captions": "preprocessed/captions.csv",
}

MASK_ROIS = {
    "both.HCP_MMP1": "preprocessed/atlas/subj-0{subject}_HCP_atlas_func_360.nii.gz",
    "both.HCP_MMP1-12dof": "preprocessed/atlas/sub-CSI{subject}_glasser_atlas_to_func_12dof.nii.gz"}

BETAS_ROIS = {
    "both.HCP_MMP1": "preprocessed/betas/sub-{subject}/sub-{subject}_hcpmmp_roi{roi}_{beta_type}.npy"
}

BOLD_DATASET = os.getenv("BOLD_DATASET")

BOLD5000_CITATION = """Chang, N., Pyles, J.A., Marcus, A. et al. 
BOLD5000, a public fMRI dataset while viewing 5000 visual images. Sci Data 6, 49 (2019). 
https://doi.org/10.1038/s41597-019-0052-3"""




def resolve_bold_path(base_dir: Optional[Path] = None) -> Path:
    """
    Resolves the BOLD5000 dataset base directory.

    Args:
        base_dir (Optional[Path]): Custom path to the dataset folder. If None, attempts to use
        the `THINGS_DATASET` environment variable.

    Returns:
        Path: The resolved path to the NSD dataset.

    Raises:
        ValueError: If the path is not provided and `THINGS_DATASET` is not set.
    """
    base_dir = base_dir or BOLD_DATASET
    if not base_dir:
        raise ValueError(
            "BOLD_DATASET dataset path not provided. "
            "Pass it as an argument or set the BOLD_DATASET environment variable."
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
    Loads a resource file for the BOLD5000 dataset as a Pandas DataFrame.

    Args:
        resource (Literal["stimulus", "images", "categories", "captions", "model-info"]): The resource to load.
        base_dir (Optional[Path]): The base directory where the dataset is located. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame containing the resource data.

    Raises:
        AssertionError: If the resource file does not exist.
    """
    base_dir = resolve_bold_path(base_dir)

    if resource not in RESOURCE_FILENAMES:
        raise ValueError(
            f"Resource {resource} not found in the NSD dataset. "
            f"Valid resources are: {list(RESOURCE_FILENAMES.keys())}"
        )

    filename = base_dir / RESOURCE_FILENAMES[resource]
    assert (
        filename.exists()
    ), f"{filename} not found! Did you complete the download and setup?"

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

    base_dir = resolve_bold_path(base_dir)
    if df_index is None:
        df = get_resource("images")
    else:
        df = df_index

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
    dataset.info.citation = " ".join(BOLD5000_CITATION.split())
    dataset.info.dataset_name = "bold5000"

    return dataset



def load_mask(
    subject: int,
    roi: Literal["both.HCP_MMP1"] = "both.HCP_MMP1",
    base_dir: Optional[Path] = None,
    return_path: bool = False,
    return_nii: bool = False,
    cast: bool = True,
):
    base_dir = resolve_bold_path(base_dir)

    if roi not in MASK_ROIS:
        raise ValueError(
            f"ROI {roi} not found in the THINGS dataset. "
            f"Valid ROIs are: {list(MASK_ROIS.keys())}"
        )

    assert subject in [1, 2, 3, 4]
    mask_name = MASK_ROIS[roi]
    mask_path = base_dir / mask_name.format(subject=subject)
    assert mask_path.exists(), f"Mask file not found at {mask_path}"
    if return_path:
        return mask_path

    image = nib.load(mask_path)
    if return_nii:
        return image
    data = image.get_fdata()
    if cast: # Cast to range 0-361
        return data.astype(np.uint16)


def get_betas_roi(
    subject: int,
    roi: Union[int, list[int]],
    atlas: Literal["both.HCP_MMP1"] = "both.HCP_MMP1",
    betas_type: Literal["A", "B", "C", "D"] = "D",
    base_dir: Optional[Path] = None,
) -> np.ndarray:
    base_dir = resolve_bold_path(base_dir)

    if atlas not in BETAS_ROIS:
        raise ValueError(
            f"ROI {atlas} not found in the THINGS dataset. "
            f"Valid ROIs are: {list(BETAS_ROIS.keys())}"
        )

    assert subject in [1, 2, 3, 4]
    if isinstance(roi, int):
        roi = [roi]
    betas = []
    for r in roi:
        betas_name = BETAS_ROIS[atlas].format(roi=r, subject=subject, beta_type=betas_type)
        betas_path = base_dir / betas_name
        assert betas_path.exists(), f"Betas file not found at {betas_path}"
        betas.append(np.load(betas_path))

    if len(betas) == 1:
        return betas[0]
    # Stack columns
    return np.hstack(betas)

def get_common_indexes(subject_x:int, subject_y:int, df_stim=None):
    if df_stim is None:
        df_stim = get_resource("stimulus")

    df_stim_x = df_stim[df_stim.subject == subject_x]
    df_stim_x = df_stim_x[["subject", "session", "bold_id", "repetition", "subject_index", "stim_source"]]
    df_stim_x = df_stim_x.rename(columns={"subject_index": "subject_index_x", "session": "session_x"})
    df_stim_y = df_stim[df_stim.subject == subject_y]
    df_stim_y = df_stim_y[["subject", "session", "bold_id", "repetition", "subject_index"]]
    df_stim_y = df_stim_y.rename(columns={"subject_index": "subject_index_y", "session": "session_y"})
    df_merge = df_stim_x.merge(df_stim_y, on=["bold_id", "repetition"], suffixes=("_x", "_y"))

    return df_merge
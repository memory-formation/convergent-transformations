"""Utility functions for working with the NSD dataset"""

import os
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING, Union, Callable
import lazy_loader as lazy
import pandas as pd
import numpy as np
from PIL import Image


nib = lazy.load("nibabel")
datasets = lazy.load("datasets")

NSD_DESCRIPTION = """The Natural Scenes Dataset (NSD) is a large-scale fMRI dataset conducted at 
    ultra-high-field (7T) strength at the Center of Magnetic Resonance Research (CMRR) at the 
    University of Minnesota. The dataset consists of whole-brain, high-resolution 
    (1.8-mm isotropic, 1.6-s sampling rate) fMRI measurements of 8 healthy adult subjects while they
    viewed thousands of color natural scenes over the course of 30-40 scan sessions. While viewing 
    these images, subjects were engaged in a continuous recognition task in which they reported 
    whether they had seen each given image at any point in the experiment. These data constitute a 
    massive benchmark dataset for computational models of visual representation and cognition, and 
    can support a wide range of scientific inquiry."""
NSD_HOMEPAGE = "https://naturalscenesdataset.org/"
NSD_CITATION = """Allen, St-Yves, Wu, Breedlove, Prince, Dowdle, Nau, Caron, Pestilli, Charest, 
    Hutchinson, Naselaris*, & Kay*. A massive 7T fMRI dataset to bridge cognitive neuroscience and 
    artificial intelligence. Nature Neuroscience (2021)."""
NSD_LICENSE = "https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions"

if TYPE_CHECKING:
    from datasets import Dataset

__all__ = [
    "resolve_nsd_path",
    "get_index",
    "load_mask",
    "load_betas",
    "get_session_indexes",
    "load_dataset",
    "get_resource",
    "get_subject_roi",
    "get_common_indexes",
]

NSD_DATASET = os.getenv("NSD_DATASET")
INDEX_FILENAMES = {
    "stimulus": "nsd_stimuli.csv",  # Relation of presentation of stimuli and images
    "mask": "nsd_masks.csv",  # Index to masks
    "images": "nsd.csv",  # Index to images
}
RESOURCE_FILENAMES = {
    "model-info": "info/models-info.csv",
    "mmp1-info": "info/mmp1-info.csv",
    "hcp": "info/hcp.csv",
    "pixtral-captions": "annotations/caption-nsd-pixtral-12b.csv",
    "coco-captions": "annotations/coco_captions.csv",
    "coco-objects": "annotations/coco_objects_annotations.csv",
    "coco-persons": "annotations/coco_persons_annotations.csv",
}


def resolve_nsd_path(base_dir: Optional[Path] = None) -> Path:
    """
    Resolves the NSD dataset base directory.

    Args:
        base_dir (Optional[Path]): Custom path to the dataset folder. If None, attempts to use
        the `NSD_DATASET` environment variable.

    Returns:
        Path: The resolved path to the NSD dataset.

    Raises:
        ValueError: If the path is not provided and `NSD_DATASET` is not set.
    """
    base_dir = base_dir or NSD_DATASET
    if not base_dir:
        raise ValueError(
            "NSD dataset path not provided. "
            "Pass it as an argument or set the NSD_DATASET environment variable."
        )
    return Path(base_dir).expanduser().resolve()


def get_index(
    file_type: Literal["stimulus", "mask", "images"], base_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Loads the index file for stimuli, masks, or images as a Pandas DataFrame.

    Args:
        file_type (Literal["stimulus", "mask", "images"]): The type of index to load.
        base_dir (Optional[Path]): The base directory where the dataset is located.
            Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame containing the index data.

    Raises:
        AssertionError: If the index file does not exist.

    Note: Maintained for backward compatibility. Use `get_resource()` instead.
    """
    return get_resource(file_type, base_dir)


def get_resource(
    resource: Union[
        Literal[
            "model-info",
            "mmp1-info",
            "hcp",
            "pixtral-captions",
            "coco-captions",
            "coco-objects",
            "coco-persons",
            "stimulus",
            "mask",
            "images",
        ],
        str,
    ],
    base_dir: Optional[Path] = None,
    return_path: bool = False,
) -> Union[pd.DataFrame, Path]:
    """
    Loads a resource file for the NSD dataset as a Pandas DataFrame.

    Args:
        resource (Literal["model-info", "mmp1-info", "pixtral-captions", "coco-captions",
        "coco-objects", "coco-persons", "stimulus", "mask", "images"]): The resource file to load.
        base_dir (Optional[Path]): The base directory where the dataset is located. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame containing the resource data.

    Raises:
        AssertionError: If the resource file does not exist.
    """
    base_dir = resolve_nsd_path(base_dir)
    FILENAMES = {**RESOURCE_FILENAMES, **INDEX_FILENAMES}
    if resource not in FILENAMES:
        raise ValueError(
            f"Resource {resource} not found in the NSD dataset. "
            f"Valid resources are: {list(FILENAMES.keys())}"
        )

    filename = base_dir / FILENAMES[resource]
    assert (
        filename.exists()
    ), f"{filename} not found! Did you complete the NSD dataset download and setup?"

    if return_path:
        return filename
    return pd.read_csv(filename)


def load_mask(
    subject: int,
    roi: Literal[
        "both.HCP_MMP1",
        "lh.HCP_MMP1",
        "rh.HCP_MMP1",
        "HCP_MMP1",
        "lh.floc-bodies",
        "rh.nsdgeneral",
        "thalamus",
        "Kastner2015",
        "lh.nsdgeneral",
        "lh.corticalsulc",
        "rh.thalamus",
        "rh.Kastner2015",
        "lh.prf-visualrois",
        "floc-bodies",
        "streams",
        "rh.floc-faces",
        "rh.prf-eccrois",
        "lh.floc-places",
        "lh.floc-words",
        "prf-eccrois",
        "rh.prf-visualrois",
        "floc-words",
        "prf-visualrois",
        "lh.MTL",
        "rh.corticalsulc",
        "MTL",
        "nsdgeneral",
        "rh.floc-places",
        "corticalsulc",
        "rh.floc-bodies",
        "rh.streams",
        "lh.floc-faces",
        "lh.prf-eccrois",
        "floc-places",
        "lh.thalamus",
        "rh.MTL",
        "lh.Kastner2015",
        "floc-faces",
        "lh.streams",
        "rh.floc-words",
    ],
    df_masks: Optional[pd.DataFrame] = None,
    base_dir: Optional[Path] = None,
    return_path: bool = False,
    return_nii: bool = False,
) -> Union["np.ndarray", Path]:
    """
    Loads an ROI mask for a specific subject.

    Args:
        subject (int): The subject ID.
        roi (str): The name of the ROI.
        df_masks (Optional[pd.DataFrame]): The masks DataFrame, if already loaded. Defaults to None.
        base_dir (Optional[Path]): The base directory where the dataset is located. Defaults to None.
        return_path (bool): Whether to return the file path of the mask instead of the data.
            Defaults to False.

    Returns:
        np.ndarray | Path: The mask data as a NumPy array or the file path to the mask if
            `return_path` is True.

    Raises:
        FileNotFoundError: If the mask file is not found.
    """

    base_dir = resolve_nsd_path(base_dir)
    df_masks = df_masks or get_index("mask")

    masks_file = df_masks.query(f"roi == '{roi}' and subject == {subject}")[
        "mask_path"
    ].values

    if not masks_file:
        available = df_masks.query(f"subject == {subject}").roi.to_list()
        raise FileNotFoundError(
            f"Mask {roi} not found for subject {subject}. Available masks are: {available}"
        )

    mask_path = base_dir / masks_file[0]

    if not mask_path.exists():
        available = df_masks.query(f"subject == {subject}").roi.to_list()
        raise FileNotFoundError(
            f"Mask {mask_path} not found. Valid masks are: {available}"
        )
    if return_path:
        return mask_path

    mask = nib.load(mask_path)

    if return_nii:
        return mask

    return mask.get_fdata()


def load_betas(
    subject: int,
    session: int,
    df_stimuli: Optional[pd.DataFrame] = None,
    base_dir: Optional[Path] = None,
    return_path: bool = False,
    return_nii: bool = False,
) -> Union["np.ndarray", Path]:
    """
    Loads the beta file for a specific subject and session.

    Args:
        subject (int): The subject ID.
        session (int): The session number.
        df_stimuli (Optional[pd.DataFrame]): The stimuli DataFrame, if already loaded.
            Defaults to None.
        base_dir (Optional[Path]): The base directory where the dataset is located.
            Defaults to None.
        return_path (bool): Whether to return the file path of the beta file instead of the data.
            Defaults to False.

    Returns:
        np.ndarray | Path: The beta data as a NumPy array or the file path to the beta file if
            `return_path` is True.

    Raises:
        FileNotFoundError: If the beta file is not found.
    """

    base_dir = resolve_nsd_path(base_dir)
    df_stimuli = df_stimuli or get_index("stimulus")

    beta = df_stimuli.query(
        f"subject == {subject} and session == {session}"
    ).filename.values[0]

    beta_path = base_dir / beta
    if not beta_path.exists():
        raise FileNotFoundError(f"Beta file {beta_path} not found")
    if return_path:
        return beta_path

    beta = nib.load(beta_path)
    if return_nii:
        return beta

    return beta.get_fdata()


def get_session_indexes(
    subject: int,
    session: int,
    df_stimuli: Optional[pd.DataFrame] = None,
    base_dir: Optional[Path] = None,
) -> list:
    """
    Retrieves the NSD IDs for a specific subject and session.

    Args:
        subject (int): The subject ID.
        session (int): The session number.
        df_stimuli (Optional[pd.DataFrame]): The stimuli DataFrame, if already loaded.
            Defaults to None.
        base_dir (Optional[Path]): The base directory where the dataset is located.
            Defaults to None.

    Returns:
        list: A list of NSD IDs corresponding to the subject and session.
    """
    base_dir = resolve_nsd_path(base_dir)
    df = df_stimuli or get_index("stimulus")

    indexes = df.query(
        f"subject == {subject} and session == {session}"
    ).nsd_id.to_list()

    return indexes


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

    base_dir = resolve_nsd_path(base_dir)
    if df_index is None:
        df = get_index("images")
    else:
        df = df_index
    #df = df_index or get_index("images")

    df["image"] = df["path"]
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
    dataset.info.description = " ".join(NSD_DESCRIPTION.split())
    dataset.info.citation = " ".join(NSD_CITATION.split())
    dataset.info.homepage = NSD_HOMEPAGE
    dataset.info.license = NSD_LICENSE
    dataset.info.dataset_name = "nsd"

    return dataset


# def get_subject_roi(subject: int, roi: int, base_dir=None) -> np.ndarray:
#     base_dir = resolve_nsd_path(base_dir)
#     folder = base_dir / "betas" / f"sub{subject:02d}"
#     filename = folder / f"sub-{subject:02d}_roi-{roi:03d}.npy"

#     if not filename.exists():
#         raise FileNotFoundError(f"File {filename} not found")

#     return np.load(filename)


def get_subject_roi(
    subject: int, roi: Union[int, list[int]], base_dir=None
) -> np.ndarray:
    base_dir = resolve_nsd_path(base_dir)
    folder = base_dir / "betas" / f"sub{subject:02d}"

    rois = [roi] if isinstance(roi, int) else roi

    betas = []
    for roi in rois:
        filename = folder / f"sub-{subject:02d}_roi-{roi:03d}.npy"
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} not found")
        betas.append(np.load(filename))

    if len(betas) == 1:
        return betas[0]

    # Betas are n_stimuli x n_voxels_roi
    betas = np.concatenate(betas, axis=1)

    return betas



def get_image(
    nsd_id: int,
    base_dir: Optional[Union[str, Path]] = None,
    df_images: Optional["pd.DataFrame"] = None,
    output_type: Literal["pil", "numpy", "path"] = "pil",
) -> Union["Image.Image", "np.ndarray", Path]:
    """
    Load an image from the NSD dataset based on its NSD ID.

    Args:

        nsd_id (int): The NSD ID of the image to load.
        base_dir (Optional[Union[str, Path]): The base directory where the dataset is located.
            Defaults to None.
        df_images (Optional[pd.DataFrame]): The images DataFrame, if already loaded.
            Defaults to None.
        output_type (Literal["pil", "numpy"]): The output type of the image. Can be either
            'pil' (PIL Image) or 'numpy' (NumPy array). Defaults to 'pil'.

    Returns:
        Image.Image | np.ndarray: The loaded image as a PIL Image or NumPy array.

    Raises:
        ValueError: If the NSD ID is not found in the index.
        AssertionError: If the output type is invalid.
    """
    assert output_type in ["pil", "numpy", "path"], "Invalid output type. Use 'pil' or 'numpy'."
    base_dir = resolve_nsd_path(base_dir=base_dir)
    if df_images is None:
        df_images = get_index("images")

    image_subset = df_images.query("nsd_id == @nsd_id")
    if image_subset.empty:
        raise ValueError(f"nsd_id {nsd_id} not found")

    image_path = base_dir / image_subset.path.values[0]

    if output_type == "path":
        return image_path
    elif output_type == "pil":
        return Image.open(image_path)
    elif output_type == "numpy":
        return np.array(Image.open(image_path))

def get_common_indexes(subject_i, subject_j, shift=0):

    df = get_index("stimulus")
    df_i = df.query(f"subject == {subject_i} and shared and exists")
    df_j = df.query(f"subject == {subject_j} and shared and exists")
    df_i = df_i[["subject", "nsd_id", "subject_index", "repetition"]]
    df_i = df_i.rename(
        columns={
            "subject": "subject_i",
            "nsd_id": "nsd_id_i",
            "subject_index": "subject_index_i",
            "repetition": "repetition_i",
        }
    )
    df_j = df_j[["subject", "nsd_id", "subject_index", "repetition"]]
    if shift:
        df_j["repetition"] = (df_j["repetition"] + shift) % 3
    df_j = df_j.rename(
        columns={
            "subject": "subject_j",
            "nsd_id": "nsd_id_j",
            "subject_index": "subject_index_j",
            "repetition": "repetition_j",
        }
    )
    df_merged = df_i.merge(
        df_j,
        left_on=["nsd_id_i", "repetition_i"],
        right_on=["nsd_id_j", "repetition_j"],
    )
    return df_merged
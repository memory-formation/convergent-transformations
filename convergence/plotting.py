import mne
import matplotlib.pyplot as plt
from typing import Optional, Literal, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    import pandas as pd

__all__ = ["plot_faverage_parcelation", "get_hcp_labels", "add_area_labels", "add_area_from_subset"]


def plot_faverage_parcelation(
    df: "pd.DataFrame",
    atlas="HCPMMP1",
    view: Optional[
        Literal["lateral", "medial", "rostral", "caudal", "dorsal", "ventral"]
    ] = "lateral",
    hemisphere: Optional[Literal["lh", "rh", "both"]] = "both",
    cortex: Literal["classic", "bone", "low_contrast"] = "high_contrast",
    surf: Literal["inflated", "pial", "white"] = "inflated",
    value_column: str = "score",
    roi_column: str = "mne_name",
    background: Union[tuple, str] = "white",
    backend: Optional[Literal["pyvistaqt", "notebook"]] = "notebook",
    cmap: str = "hot",
    filename: Optional[Union["Path", str]] = None,
    normalize=True,
    default_value: float = 0,
    default_color: Optional[str] = None,
    borders: Union[bool, float] = False,
    close: Optional[bool] = None,
    subjects_dir: Optional[Union["Path", str]] = None,
    fetch_parcelation: bool = False,
    verbose: bool = False,
    size=(800, 600),
    smooth=0,
):
    """
    Plot a parcellation on the fsaverage brain surface using MNE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the ROI names and values for coloring. Must include
        columns specified by `value_column` and `roi_column`.
    atlas : str, optional
        The parcellation atlas to use. Default is "HCPMMP1".
    view : Optional[Literal["lateral", "medial", "rostral", "caudal", "dorsal", "ventral"]], optional
        The view of the brain surface to display. Default is "lateral".
    hemisphere : Optional[Literal["lh", "rh", "both"]], optional
        Hemisphere to visualize. Options are "lh" (left), "rh" (right), or "both". Default is "both".
    cortex : Literal["classic", "bone", "low_contrast", "high_contrast"], optional
        The shading style of the cortical surface. Default is "high_contrast".
    surf : Literal["inflated", "pial", "white"], optional
        The surface type to visualize. Options include "inflated", "pial", or "white". Default is "inflated".
    value_column : str, optional
        The column in `df` that contains the values for coloring. Default is "score".
    roi_column : str, optional
        The column in `df` that contains the ROI names matching the atlas. Default is "mne_name".
    background : Union[tuple, str], optional
        Background color of the visualization. Can be a color string or an RGB tuple. Default is "white".
    backend : Optional[Literal["pyvistaqt", "notebook"]], optional
        The backend to use for 3D visualization. Default is "notebook".
    cmap : str, optional
        The colormap to use for displaying values. Default is "hot".
    filename : Optional[Union[Path, str]], optional
        If specified, saves the output as an image file. Default is None.
    normalize : Union[bool, tuple], optional
        If True, normalizes the values in `df[value_column]` between 0 and 1. If a tuple is provided,
        it defines the (vmin, vmax) range for normalization. Default is True.
    default_value : float, optional
        The value to assign if an ROI is not found in `df`. Default is 0.
    borders : Union[bool, float], optional
        If True or a float, adds borders around each ROI. If a float, specifies the border thickness. Default is False.
    close : Optional[bool], optional
        Whether to close the Brain window after plotting. If None, the window is closed if `filename` is specified. Default is None.
    subjects_dir : Optional[Union[Path, str]], optional
        The directory containing the FreeSurfer `subjects` data. If None, uses the default MNE subjects directory.
    fetch_parcelation : bool, optional
        If True, fetches the HCP-MMP parcellation if not already downloaded. Default is False.

    Returns
    -------
    mne.viz.Brain
        The Brain object used for the visualization.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "mne_name": ["L_V1_ROI-lh", "R_V2_ROI-rh"],
    ...     "score": [0.8, 0.6]
    ... })
    >>> brain = plot_faverage_parcelation(df, view="lateral", hemisphere="both")
    >>> brain.show()

    Notes
    -----
    - The function relies on MNE and Matplotlib for visualization.
    - Ensure that the HCP-MMP parcellation is available in your FreeSurfer subjects directory.
    """

    if backend:
        mne.viz.set_3d_backend(backend, verbose=False)

    if not subjects_dir:
        subjects_dir = mne.datasets.sample.data_path() / "subjects"
    if fetch_parcelation:
        mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=False)

    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc=atlas,
        hemi=hemisphere,
        subjects_dir=subjects_dir,
        verbose=False,
        surf_name=surf,
    )

    # Initialize the Brain object for visualization
    Brain = mne.viz.get_brain_class()
    brain = Brain(
        "fsaverage",
        hemi=hemisphere,
        surf=surf,
        subjects_dir=subjects_dir,
        cortex=cortex,
        background=background,
        size=size,
        show=False,
    )
    cmap = plt.get_cmap(cmap)
    df_aux = df[[roi_column, value_column]].copy()
    if isinstance(normalize, tuple) and len(normalize) == 2:
        vmin, vmax = normalize
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        df_aux["norm_score"] = norm(df_aux[value_column])
    elif normalize:
        norm = plt.Normalize(vmin=df_aux[value_column].min(), vmax=df_aux[value_column].max())
        df_aux["norm_score"] = norm(df_aux[value_column])
    else:
        df_aux["norm_score"] = df_aux[value_column]

    # Plot the HCP-MMP labels with colors based on DataFrame values
    for label in labels:
        # Find the corresponding value in the DataFrame

        match = df_aux[df_aux[roi_column] == label.name]
        if not match.empty:
            if verbose:
                print("Found", label.name)
            value = match["norm_score"].values[0]
            color = cmap(value)
            if smooth:
                label = label.smooth(subjects_dir=subjects_dir, smooth=5, verbose=False)
            brain.add_label(label, color=color, borders=borders)
        else:
            if verbose:
                print("Not found", label.name)

            if default_value is not None:
                color = cmap(default_value)
            elif default_color is not None:
                color = default_color
            else:
                color = None
            if color is not None:
                if smooth:
                    label = label.smooth(subjects_dir=subjects_dir, smooth=5, verbose=False)
                brain.add_label(label, color=color, borders=borders)
            
            value = default_value

        
        

    if view:
        brain.show_view(view)
    if filename is not None:
        brain.save_image(filename, mode="rgba")
        if close is None:
            close = True
    if close:
        brain.close()
    return brain


def get_hcp_labels(atlas="HCPMMP1", hemisphere="both", surf="inflated", subjects_dir=None):

    if subjects_dir is None:
        subjects_dir = mne.datasets.sample.data_path() / "subjects"

    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc=atlas,
        hemi=hemisphere,
        subjects_dir=subjects_dir,
        verbose=False,
        surf_name=surf,
    )

    return labels


def add_area_from_subset(brain, subset, color, labels=None, alpha: float=1, borders:bool=True):
    if subset.empty:
        return
    if labels is None:
        labels = get_hcp_labels()
    areas = subset.mne_name.to_list()
    selected_labels = list(filter(lambda x: x.name in areas, labels))
    if len(selected_labels) > 1:
        combined_labels = sum(selected_labels[1:], selected_labels[0])
        brain.add_label(combined_labels, borders=borders, color=color, alpha=alpha)
    elif len(selected_labels) == 1:
        brain.add_label(selected_labels[0], borders=borders, color=color, alpha=alpha)


    

def add_area_labels(brain, hcp, area_ids=None, color=None, alpha: float = 1, hemispheres=["lh", "rh"]):
    if area_ids is None:
        area_ids = hcp.area_id.unique()

    labels = get_hcp_labels()
    global_color = color

    for area_id in area_ids:
        for hemi in hemispheres:
            sub_hcp = hcp.query(f"area_id=={area_id} and hemisphere=='{hemi}'")
            if sub_hcp.empty:
                continue
            if global_color is None:
                color = sub_hcp.area_color.tolist()[0]
            else:
                color = global_color
            add_area_from_subset(brain, sub_hcp, color, labels, alpha=alpha)

def save_brain_views(
    df,
    name,
    hemispheres=["lh", "rh"],
    views=["lateral", "medial", "rostral", "caudal", "dorsal", "ventral"],
    hcp=None,
    area_ids=None,
    size=(2*800, 2*600),
    adjust=False,
    formats=["png"],
    **kwargs,
):
    for hemi in hemispheres:
        brain = plot_faverage_parcelation(df, hemisphere=hemi, size=size, **kwargs)
        if hcp is not None:
            add_area_labels(brain, hcp=hcp, area_ids=area_ids, hemispheres=[hemi])

        for view in views:
            if view == 'lateral' and adjust:
                if hemi == 'lh':
                    brain.show_view("lateral", azimuth=20, elevation=-100)
                else:
                    brain.show_view("lateral", azimuth=-20, elevation=100)
            elif view == 'medial' and adjust:
                if hemi == 'lh':
                    brain.show_view("medial",  azimuth=-20, elevation=100)
                else:
                    brain.show_view("medial",  azimuth=20, elevation=-100)
            else:
                brain.show_view(view)
            for fmt in formats:
                brain.save_image(f"{name}_{hemi}_{view}.{fmt}", mode="rgba")
        brain.close()

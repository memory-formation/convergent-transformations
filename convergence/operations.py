from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def interpolate_alignment_series(
    df: pd.DataFrame,
    groups: list[str] = ["metric", "subject", "model", "atlas", "session", "roi"],
    layer_column: str = "layer",
    score_column: str = "score",
    interp_scores_column: str = "scores",
    n_values: Optional[int] = None,
    progress_bar: bool = True,
):

    if n_values is None:
        n_values = df.layer.max()

    def interpolate_series(series):
        series = series.sort_values(layer_column)
        x_orig = np.linspace(0, 1, len(series))
        y_orig = series[score_column].values
        x_interp = np.linspace(0, 1, n_values)
        y_interp = np.interp(x_interp, x_orig, y_orig)
        return y_interp

    dfs = []
    models = list(df.model.unique())
    if progress_bar:
        models = tqdm(models, leave=False)

    for model in models:
        df_model = df.query(f"model==@model and {layer_column} >= 0")
        grouped = df_model.groupby(groups, observed=True)
        df_interp = grouped.apply(interpolate_series).reset_index()
        df_interp = df_interp.rename(columns={0: interp_scores_column})
        dfs.append(df_interp)
    dfs = pd.concat(dfs)
    dfs = dfs.sort_values(groups).reset_index(drop=True)
    return dfs


def name_to_mne(name, atlas):
    name = name.replace(" ", "")
    name = name.replace("7Pl", "7PL")
    if atlas == "HCP_MMP1":
        return name
    if atlas == "rh.HCP_MMP1":

        return f"R_{name}_ROI-rh"
    if atlas == "lh.HCP_MMP1":
        return f"L_{name}_ROI-lh"
    
    return None

def name_to_mne_pandas(df, name_column='name', atlas_column='atlas'):
    return df.apply(lambda row: name_to_mne(row[name_column], row[atlas_column]), axis=1)   

def unpack_scores(df):
    dfs = []
    df = df.copy()
    scores = df.scores
    info = df.drop(columns=["scores"])
    
    for t in range(len(scores.values[0])):
        df = info.copy()
        df["score"] = scores.str[t]
        df["layer"] = t
        dfs.append(df)

    df = pd.concat(dfs)
    #df = df.sort_values(by=["subject", "roi", "modality"]).reset_index(drop=True)
    return df
import pandas as pd


visual_cortex = ["V1", "V2", "V3", "V4"]  # EVC
ventral_cluster = ["VMV1", "VMV2", "VMV3", "PHA1", "PHA2", "PHA3"]  # Ventral (PPA)
dorsal_cluster = ["MT", "MST", "FST", "V4t", "TPOJ2", "TPOJ3"]  # Lateral/Dorsal LOTC


def add_cluster(data):

    def get_cluster(roi):
        if roi in visual_cortex:
            return "Visual Cortex"
        elif roi in ventral_cluster:
            return "Ventral Cluster"
        elif roi in dorsal_cluster:
            return "Dorsal Cluster"
        else:
            return "Other"

    data["cluster"] = data["name"].apply(get_cluster)

    return data


def process_intersubject_rois(df, hcp_filename, top=10):
    # Load cross subject data
    df = df.rename(columns={"similarity": "score"})
    df = df.query("score < 1 and score > -1 and roi_x == roi_y and subject_i != subject_j").copy()
    df = df.groupby(["roi_x", "subject_i"]).aggregate({"score": "mean"}).reset_index()
    df = df.rename(columns={"roi_x": "roi", "subject_i": "subject"})

    # HCP
    hcp = pd.read_csv(hcp_filename)
    hcp.name = hcp.name.replace({"H": "Hipp"})
    hcp = hcp[["roi", "name", "roi_order", "area_color", "area_id", "area"]]

    # Get areas with highest intersubject alignment
    df_top = df.groupby("roi").aggregate({"score": "mean"}).reset_index()
    df_top = df_top.merge(hcp[["roi", "name", "area_id", "area"]], on="roi")
    df_top = (
        df_top.groupby(["area", "area_id"])
        .aggregate({"score": "mean"})
        .reset_index()
        .sort_values("score", ascending=False)
    )
    selected_areas = df_top.head(top).sort_values("area_id").area_id.tolist()

    df = df.merge(hcp[["roi", "name", "area_id", "area", "roi_order", "area_color"]], on="roi")
    df = df.sort_values(["area_id", "roi_order"]).reset_index(drop=True)

    df_g_filtered = df.query(
        "area_id in @selected_areas"
    ).copy()  # and roi in @very_significant_rois").copy()
    df_g_filtered = df_g_filtered.sort_values(["area_id", "roi_order"]).reset_index(drop=True)

    return df_g_filtered


def _load_filename(filename):
    if str(filename).endswith(".parquet"):
        return pd.read_parquet(filename)
    elif str(filename).endswith(".csv"):
        return pd.read_csv(filename)
    else:
        raise ValueError(
            f"Unsupported file format for {filename}. Supported formats are .parquet and .csv."
        )


def proccess_alignment(
    models_filename,
    models_alignment_filename,
    subject_alignment_filename,
    pvalues_filename,
    hcp_filename,
    group_subject=True,
):
    models_info = pd.read_csv(models_filename)
    models_info = models_info[["model_name", "modality"]].rename(columns={"model_name": "model"})
    df_models = _load_filename(models_alignment_filename)
    # df_models = load_filename("models", joined=joined)
    df_models = df_models.rename(columns={"similarity": "score"})
    df_models = df_models.query("score < 1 and score > -1")  # Not needed. No one is out of bounds
    df_models["max_layer"] = df_models.groupby(["model"], observed=True).layer.transform("max")
    df_models["depth"] = df_models["layer"] / df_models["max_layer"]
    df_models["abs_score"] = df_models["score"].abs()
    df_models = df_models.sort_values("abs_score", ascending=False).drop_duplicates(
        ["roi", "model", "subject", "session"], keep="first"
    )  # Get the best score for each roi, model, subject, session

    df_models = (
        df_models.groupby(["roi", "model", "subject"], observed=True)
        .aggregate({"score": "mean", "depth": "mean"})
        .reset_index()
    )
    df_models = df_models.merge(models_info, on="model")

    if group_subject:
        df_models = (
            df_models.groupby(["roi", "modality"])
            .aggregate({"score": "mean", "depth": "mean"})
            .reset_index()
        )
    else:
        df_models = (
            df_models.groupby(["roi", "modality", "subject"])
            .aggregate({"score": "mean", "depth": "mean"})
            .reset_index()
        )
    df_models_language = df_models.query("modality == 'language'")
    df_models_vision = df_models.query("modality == 'vision'")

    df_intersubject = _load_filename(subject_alignment_filename)
    # df_intersubject = load_filename("cross_subject", shift=1, joined=joined)
    df_intersubject = process_intersubject_rois(
        df_intersubject, hcp_filename=hcp_filename, top=1000
    )
    groups = ["roi"] if group_subject else ["roi", "subject"]
    df_intersubject = df_intersubject.groupby(groups).score.mean().reset_index()
    df_intersubject = df_intersubject.rename(columns={"score": "intersubject_rsa"})

    df_models_language = df_models_language.rename(
        columns={"score": "language_rsa", "depth": "language_depth"}
    )
    df_models_language = df_models_language[groups + ["language_rsa", "language_depth"]]
    df_models_vision = df_models_vision.rename(
        columns={"score": "vision_rsa", "depth": "vision_depth"}
    )
    df_models_vision = df_models_vision[groups + ["vision_rsa", "vision_depth"]]
    df_comparison = df_intersubject.merge(df_models_language, on=groups).merge(
        df_models_vision, on=groups
    )

    p_values = _load_filename(pvalues_filename)
    p_values_intersubject = p_values.query("comparison == 'intersubject'")
    p_values_intersubject = p_values_intersubject[
        ["roi", "pvalue_fdr_bh", "null_mean", "null_std", "apa_star"]
    ].rename(
        columns={
            "pvalue_fdr_bh": "intersubject_p_value",
            "null_mean": "intersubject_null_mean",
            "null_std": "intersubject_null_std",
            "apa_star": "intersubject_apa_star",
        }
    )
    df_comparison = df_comparison.merge(p_values_intersubject, on="roi", how="left")

    hcp = pd.read_csv(hcp_filename)
    hcp = hcp[["roi", "name", "area_color", "area_id", "area", "mne_name", "roi_order"]]
    df_comparison = df_comparison.merge(hcp, on="roi", how="left")

    df_pvalues_vision = p_values.query("comparison == 'vision'")[
        ["roi", "pvalue_fdr_bh", "null_mean", "null_std", "apa_star"]
    ].rename(
        columns={
            "pvalue_fdr_bh": "vision_p_value",
            "null_mean": "vision_null_mean",
            "null_std": "vision_null_std",
            "apa_star": "vision_apa_star",
        }
    )
    df_pvalues_language = p_values.query("comparison == 'language'")[
        ["roi", "pvalue_fdr_bh", "null_mean", "null_std", "apa_star"]
    ].rename(
        columns={
            "pvalue_fdr_bh": "language_p_value",
            "null_mean": "language_null_mean",
            "null_std": "language_null_std",
            "apa_star": "language_apa_star",
        }
    )
    df_comparison = df_comparison.merge(df_pvalues_vision, on="roi", how="left")
    df_comparison = df_comparison.merge(df_pvalues_language, on="roi", how="left")

    return df_comparison


def add_hcp_names(df, hcp_filename, columns=(("roi", "name"), ("roi_x", "name_x"), ("roi_y", "name_y")), hcp_columns=["roi", "name"]):
    hcp = pd.read_csv(hcp_filename)
    hcp = hcp[hcp_columns]
    for roi_col, name_col in columns:
        if roi_col not in df.columns: continue
        hcp_renamed = hcp.rename(columns={"roi": roi_col, "name": name_col})
        df = df.merge(hcp_renamed, on=roi_col, how="left")
    return df


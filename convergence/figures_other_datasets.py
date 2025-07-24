from typing import Literal
import pandas as pd

def load_alignment(
        filenames,
        hcp_filename,
        subset: Literal["all", "coco", "scenes", "imagenet"] = "all", 
        joined: bool=True, 
        order_column="abs_similarity"):
    join_hemispheres = "joined" if joined else "separated"
    intersubject_file = filenames[subset]["subject"][join_hemispheres]
    df_intersubject = pd.read_parquet(intersubject_file)
    df_intersubject = df_intersubject.query("roi_x == roi_y and subject_i != subject_j").groupby("roi_x").similarity.mean()
    df_intersubject = df_intersubject.reset_index()
    df_intersubject = df_intersubject.rename(columns={"roi_x": "roi", "similarity": "rsa_intersubject"})
    df_intersubject

    model_file = filenames[subset]["model"][join_hemispheres]
    df_model = pd.read_parquet(model_file)
    # Max layer is max of column layer grouped by model column
    df_model["max_layer"] = df_model.groupby("model")["layer"].transform("max")
    df_model["abs_similarity"] = df_model["similarity"].abs()
    # # Sort by absolute similarity
    df_model = df_model.sort_values(order_column, ascending=False)
    df_model = df_model.drop_duplicates(subset=["roi", "model", "session", "subject"], keep="first")
    df_model["depth"] = df_model["layer"] / df_model["max_layer"]
    df_model = df_model.groupby(["roi", "model", "subject"]).aggregate({
        "similarity": "mean",
        "depth": "mean",
    }).reset_index()
    df_model = df_model.groupby(["roi", "model"]).aggregate({
        "similarity": "mean",
        "depth": "mean",
    }).reset_index()
    df_model["modality"] = df_model.model.str.startswith("vit").map({True: "Vision", False: "Language"})

    df_model = df_model.groupby(["roi", "modality"]).aggregate({
        "similarity": "mean",
        "depth": "mean",
    }).reset_index()

    df_model_language = df_model.query("modality == 'Language'").copy()
    df_model_language = df_model_language.rename(columns={
        "similarity": "rsa_language",
        "depth": "depth_language"
        }).drop(columns=["modality"])
    df_model_vision = df_model.query("modality == 'Vision'").copy()
    df_model_vision = df_model_vision.rename(columns={
        "similarity": "rsa_vision",
        "depth": "depth_vision"
        }).drop(columns=["modality"])

    df_joined = df_intersubject.merge(df_model_language, on="roi", how="inner", validate="1:1")
    df_joined = df_joined.merge(df_model_vision, on="roi", how="inner", validate="1:1")
    hcp = pd.read_csv(hcp_filename)
    hcp = hcp[["roi", "name", "area_id", "area", "area_color", "roi_order", "mne_name"]]
    df_joined = df_joined.merge(hcp, on="roi", how="inner")
    df_joined = df_joined.sort_values("roi").reset_index(drop=True)
    return df_joined
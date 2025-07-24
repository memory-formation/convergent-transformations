"""
Helper function of the notebook kmcca projections related with plotting the results of controlled RSA.
"""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from .figures import add_pvalue_bracket

def unify_controlled_rsa_data(
    filename_hcp: Path,
    filename_subject_controlled: Path,
    filename_model_controlled: Path,
    models_info_filename: Path,
    cluster_rois: dict,
    group_cluster: bool = False,
    group_subjects: bool = False,
):
    df_hcp = pd.read_csv(filename_hcp)
    df_hcp = df_hcp[["roi", "name", "mne_name", "area", "area_id"]]
    map_cluster_rois = {v: k for k, values in cluster_rois.items() for v in values}

    # Load intersubject controlled data
    df_controled_subjects = []
    for modality, query in [
        ("Withinsubject", "subject_i==subject_j"),
        ("Intersubject", "subject_i!=subject_j"),
    ]:
        df_controled = pd.read_parquet(filename_subject_controlled)
        df_controled = df_controled.query(f"roi_x == roi_y and {query}")
        df_controled = df_controled.groupby(
            ["roi_x", "subject_i", "rep_shift", "control"], observed=True
        )
        df_controled = df_controled.similarity.mean().reset_index()
        df_controled = df_controled.merge(df_hcp, left_on="roi_x", right_on="roi", how="inner")
        df_controled = df_controled.drop(columns=["roi_x"])
        df_controled["cluster"] = df_controled["name"].map(map_cluster_rois).fillna("Other")
        df_controled = df_controled.rename(columns={"subject_i": "subject"})
        df_controled["modality"] = modality
        df_controled_subjects.append(df_controled)

    df_controled = pd.concat(df_controled_subjects, ignore_index=True)

    # Sort columns
    columns = [
        "roi",
        "subject",
        "control",
        "modality",
        "similarity",
        "name",
        "cluster",
        "mne_name",
        "area",
        "area_id",
        "rep_shift",
    ]
    df_controled = df_controled[columns]
    # Load model controlled data
    df_model_controled = pd.read_parquet(filename_model_controlled)
    models_info = pd.read_csv(models_info_filename)
    # Take max across model layers, mean across modality
    df_model_controled = (
        df_model_controled.groupby(["model", "roi", "subject", "control"], observed=True)
        .similarity.max()
        .reset_index()
    )
    df_model_controled = df_model_controled.merge(
        models_info[["model_name", "modality"]], left_on="model", right_on="model_name", how="inner"
    )
    df_model_controled = (
        df_model_controled.groupby(["roi", "modality", "control", "subject"], observed=True)
        .similarity.mean()
        .reset_index()
    )
    df_model_controled = df_model_controled.merge(df_hcp, on="roi", how="inner")
    df_model_controled["cluster"] = df_model_controled["name"].map(map_cluster_rois).fillna("Other")
    df_model_controled["rep_shift"] = -1
    df_model_controled["modality"] = df_model_controled["modality"].str.capitalize()
    df_model_controled = df_model_controled[columns]
    assert len(df_model_controled) == len(
        df_controled
    ), f"{len(df_model_controled)} != {len(df_controled)}"

    # Join in a single table
    df_controled = pd.concat([df_controled, df_model_controled], ignore_index=True)
    df_controled.modality = df_controled.modality.astype(str)

    if group_cluster:
        df_controled = df_controled.query("cluster!='Other'").groupby(
            ["modality", "cluster", "control", "subject"], observed=True
        )
        df_controled = df_controled.similarity.mean().reset_index()
    

    if group_subjects:
        columns = [c for c in df_controled.columns if c != "subject" and c != "similarity"]
        df_controled = df_controled.groupby(columns, observed=True).similarity.mean().reset_index()

    return df_controled

def obtain_modality_main_controls(df_controled_clusters, modality):
    df_plot_controled_clusters = df_controled_clusters.query("""modality == @modality and 
                                (((cluster == 'Visual Cortex' and control == 'visual_cortex_cca_1')
                                or (cluster == 'Ventral Hub' and control == 'ventral_hub_cca_1')
                                or (cluster == 'LOTC Hub' and control == 'dorsal_hub_cca_1')
                                ) or (control == 'uncontrolled')
                                )
                                """.replace("\n", " ")
    ).copy()

    assert len(df_plot_controled_clusters[["cluster", "control"]].drop_duplicates()) == 6

    # Rename controlled variables to be more readable
    variable_map = {"visual_cortex_cca_1": "KMCCA 1 dimension",
        "ventral_hub_cca_1": "KMCCA 1 dimension",
        "dorsal_hub_cca_1": "KMCCA 1 dimension",
        "uncontrolled": "None (Baseline)"
    }
    df_plot_controled_clusters.control = df_plot_controled_clusters.control.map(variable_map)

    # Compare the controlled cluster with the baseline (uncontrolled) cluster
    df_plot_controled_clusters_controlled = df_plot_controled_clusters.query("control != 'None (Baseline)'")
    df_plot_controled_clusters_base = df_plot_controled_clusters.query("control == 'None (Baseline)'")
    df_plot_controled_clusters_base = df_plot_controled_clusters_base.drop(columns=["control"])
    df_plot_controled_clusters_base = df_plot_controled_clusters_base.rename(columns={"similarity": "rsa"})
    df_plot_controled_clusters_controlled = df_plot_controled_clusters_controlled.rename(columns={"similarity": "partial_rsa"})
    df_comparison_control = df_plot_controled_clusters_controlled.merge(
        df_plot_controled_clusters_base,
        on=["modality", "cluster", "subject"]
    )
    
    comparisons = df_comparison_control.groupby(["cluster", "modality"]).apply(lambda x: ttest_rel(x["rsa"], x["partial_rsa"]), include_groups=False).reset_index()
    comparisons["p_value"] = comparisons[0].apply(lambda x: x.pvalue)
    comparisons["t_statistic"] = comparisons[0].apply(lambda x: x.statistic)
    comparisons = comparisons.drop(columns=[0])
    comparisons["p_value_corrected"] = multipletests(comparisons["p_value"], method="bonferroni")[1]
    comparisons["apa_star"] = comparisons["p_value_corrected"].apply(apa_star)

    return df_plot_controled_clusters, comparisons, df_comparison_control


def plot_clusters_controled(
    df_plot_controled_clusters,
    comparisons,
    modality="Intersubject",
    cluster_order = ['Visual Cortex', 'Ventral Hub', 'LOTC Hub'],
    control_order = ['None (Baseline)', 'KMCCA 1 dimension'],
    h3=0.04,
    h4=0,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = None

    sns.boxplot(data=df_plot_controled_clusters, x='cluster', y='similarity', hue='control', ax=ax, order=cluster_order, hue_order=control_order,  gap=0.23, showfliers=False)
    # Add points with jitter
    sns.stripplot(data=df_plot_controled_clusters, x='cluster', y='similarity', hue='control', ax=ax, order=cluster_order, 
        hue_order=control_order, alpha=1, marker="o", size=5, edgecolor=(0.24, 0.24, 0.24), linewidth=0.5, jitter=0.03, legend=False, dodge=True)
                
    for i, cluster in enumerate(cluster_order):
        df_subset_controled = df_plot_controled_clusters.query("cluster == @cluster and control !='None (Baseline)'")
        df_subset_uncontroled = df_plot_controled_clusters.query("cluster == @cluster and control =='None (Baseline)'")
        am = df_subset_uncontroled.similarity.mean()
        bm = df_subset_controled.similarity.mean()
        a = df_subset_uncontroled.similarity.median()
        b = df_subset_controled.similarity.median()
        mininimun = df_subset_controled.similarity.min()
        abs_diff = (bm-am)
        rel_diff = abs_diff / max(am, bm)
        # print(f"{cluster} - RSA vs Partial RSA. Absolute difference: {abs_diff:.2f}, Relative difference: {100*rel_diff:.1f}%.")
        #print(f"{cluster} - Absolute difference: {
        c = 0.42 + i
        w = 0.02
        w2 = 0.15
        ax.plot([c, c], [a, b], color=(0.24, .24, .24), lw=1)
        ax.plot([c-w, c], [a, a], color=(0.24, .24, .24), lw=1)
        ax.plot([c-w, c], [b, b], color=(0.24, .24, .24), lw=1)
        # ax4.plot([i-w2, i+w2], [a, b], color=(0.24, .24, .24), lw=1, ls='-')
        
        t = 0 if i!=2 else h4
        ax.text(i,mininimun -h3-t, f"$\\Delta\\rho = {abs_diff:.2f}$\n(${100*rel_diff:.1f} \%$)", ha='center', va='center', fontsize=10)
        #ax4.text(c+ 0.02, (a+b)/2, f"$\\Delta\\rho$", ha='left', va='center', fontsize=10)
    ax.set_ylim(0.0, None)
    # lower center
    ax.legend(loc="lower center", title="")
    ax.set_title("Partial RSA (Controlled KMCCA)", fontsize="large")
    ax.set_xlabel("")
    ax.set_ylabel(f"{modality} Alignment (Partial RSA Person's $\\rho$)")
    #ax.set_ylim(0,0.23)
    w = 0.18
    h = 0.005
    h2 = 0.01
    # Add p-value brackets
    clusters_max = df_plot_controled_clusters.groupby("cluster").similarity.max()
    for i, cluster in enumerate(cluster_order):
        apa_star_v = comparisons.set_index("cluster").loc[cluster].apa_star
        add_pvalue_bracket(ax=ax, x1=i-w, x2=i+w, y=clusters_max.loc[cluster]+h2, height=h, text=apa_star_v)
    sns.despine(ax=ax)
    return fig, ax

def apa_star(p_value: float) -> str:
    """Return a string with the APA star notation for a given p-value."""
    if p_value < 0.0001:
        return "****"
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "n.s."  # Not significant

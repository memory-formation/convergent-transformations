import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path


def setup_matplotlib_fonts():
    # Change font to arial and avoid rasterization in SVG and PDF
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42

def plot_boxplot_rois(
    df,
    ax,
    order,
    hue_order,
    palette,
    title=None,
    legend=True,
    fontsize=7,
    legend_fontsize=9,
    vmax=0.26,
    strip_size=3,
    legend_kwargs={},
    **kwargs,
):
    sns.boxplot(
        data=df,
        x="name",
        hue="area",
        y="score",
        ax=ax,
        order=order,
        hue_order=hue_order,
        palette=palette,
        legend=legend,
        zorder=100,
        showfliers=False,
        **kwargs,
    )

    # Plot the individual points with jitter
    sns.stripplot(
        data=df,
        x="name",
        hue="area",
        y="score",
        ax=ax,
        order=order,
        hue_order=hue_order,
        palette=palette,
        # jitter=False,
        # dodge=True,
        linewidth=0.5,
        edgecolor=(0.24, 0.24, 0.24),
        zorder=200,
        marker="o",
        s=strip_size,
        legend=False,
    )

    if legend:
        ax.legend(loc="upper right", ncol=2, fontsize=legend_fontsize, **legend_kwargs)

    ax.set_ylabel("Alignment (CKA)")
    ax.set_xlabel("")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, 0))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=fontsize)
    # ax.grid(axis='y', linestyle='--', alpha=0.5)
    for v in range(1, 7):
        color = "maroon" if v == 0 else "gray"
        ax.axhline(v * 0.05, color=color, linestyle="--", lw=0.5, zorder=-1)

    s = 0
    counts = df.drop_duplicates("name").area.value_counts().to_dict()
    for area in hue_order:
        s = s + counts[area]
        ax.axvline(s - 0.5, color="gray", ls="--", alpha=0.5, lw=0.3, zorder=-10)

    eps = 1
    ax.set_xlim(-eps, len(order) + eps)
    ax.set_ylim(-0.01, vmax)
    sns.despine(ax=ax)
    if title is not None:
        # Move into the plot
        ax.set_title(title, fontsize=14)


def plot_cbar(
    figsize=None,
    cmap="viridis",
    vmin=0,
    vmax=0.2,
    horizontal=False,
    title="",
    percent=True,
    locator=0.05,
    rotation=-90,
    labelpad=20,
    fontsize=12,
    **kwargs,
):

    if figsize is None:
        figsize = (6, 0.4) if horizontal else (0.4, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Create a colormap normalization
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Create a scalar mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for ScalarMappable

    # Add the colorbar
    orientation = "horizontal" if horizontal else "vertical"
    cbar_instance = fig.colorbar(sm, cax=ax, orientation=orientation, **kwargs)
    cbar_instance.set_label(title, rotation=rotation, labelpad=labelpad, fontsize=fontsize)

    if percent:
        ticker = mticker.PercentFormatter(xmax=1, decimals=0)
        cbar_instance.formatter = ticker
    if locator:
        cbar_instance.locator = mticker.MultipleLocator(base=locator)

    return fig, ax


# def plot_comparison(
#     df,
#     x,
#     y,
#     xlabel="",
#     ylabel="",
#     threshold=0.038,
#     add_identity=False,
#     ylim=None,
#     xlim=None,
#     custom={},
#     scatter_kws={},
#     fontsize=6,
#     legend=False,
# ):
#     areas = df[["area_id", "area", "area_color"]].sort_values("area_id").drop_duplicates("area_id")

#     hue_order = list(areas.area.tolist())
#     palette = list(areas.area_color.tolist())

#     g = sns.jointplot(
#         data=df,
#         x=x,
#         y=y,
#         kind="scatter",
#         marginal_kws=dict(bins=50, color="gray"),
#         ylim=ylim,
#         xlim=xlim,
#     )
#     ax = g.ax_joint
#     ylims = ax.get_ylim()
#     xlims = ax.get_xlim()
#     ax.clear()
#     ax.set_xlim(xlims)
#     ax.set_ylim(ylims)
#     sns.scatterplot(
#         data=df,
#         x=x,
#         y=y,
#         hue="area",
#         palette=palette,
#         hue_order=hue_order,
#         ax=ax,
#         legend=legend,
#         zorder=10,
#         **scatter_kws,
#     )
#     if add_identity:
#         ax.plot([0, 1], [0, 1], ls="--", lw=0.5, color="maroon", zorder=-10)
#     ax.xaxis.set_major_formatter(mticker.PercentFormatter(1, 0))
#     ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, 0))
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     # Add roi names as text labels
#     for _, row in df.query(f"{y}>{threshold} or {x}>{threshold}").iterrows():
#         shift_y = 0.004
#         shift_x = 0
#         name = row["name"]
#         if name in custom:
#             shift_x, shift_y = custom[name]
#         ax.text(
#             row[x] + shift_x,
#             row[y] + shift_y,
#             name,
#             ha="center",
#             va="center",
#             fontsize=fontsize,
#             font="Arial",
#             color="black",
#             zorder=20,
#         )
#     return g



def add_pvalue_bracket(
    ax, x1, x2, y, height, color=(0.24, 0.24, 0.24), linewidth=1, text=None, y_text=None, **kwargs
):
    """
    Draws a bracket (like those used to denote p-values) on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the bracket.
    x1, x2 : float
        The x-coordinates between which the bracket is drawn.
    y : float
        The y-coordinate at which the bracket starts.
    height : float
        The vertical distance (height) of the bracket.
    color : str, optional
        The color of the lines.
    linewidth : float, optional
        The width of the lines.
    """
    # Draw the bracket:
    #   Vertical line up from (x1, y) to (x1, y+height)
    #   Horizontal line from (x1, y+height) to (x2, y+height)
    #   Vertical line down from (x2, y+height) to (x2, y)
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color=color, linewidth=linewidth)
    if text:
        if not y_text:
            y_text = y + height
        ax.text((x1 + x2) * 0.5, y_text, text, ha="center", **kwargs)


def plot_comparison(data, x, y, ax, palette, hue="area"):

    # Marker style
    style = {
        'Ventral Cluster': 's', # Square
        'Visual Cortex': '^', # Triangle
        'Dorsal Cluster': 'D',
        'Other': 'o' # Normal Dot
    }
    sns.scatterplot(
        data=data.query("not top_area"),
        x=x,
        y=y,
        color="white",
        ax=ax,
        edgecolor=(0.24, 0.24, 0.24),
        zorder=-5,
        alpha=0.8,
    )

    sns.scatterplot(
        data=data.query("top_area"),
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        ax=ax,
        #s=100,
        edgecolor=(0.24, 0.24, 0.24),
        #linewidth=0.5,
        legend=False,
        style="cluster",
        markers=style,
    )


    sns.despine(ax=ax)
    return ax

def plot_cbar_set(
    title: str,
    filename: Path,
    vmin=-0.2,
    vmax=-0.2,
    cmap="RdBu_r",
    locator=0.1,
    figsize=(0.2, 6),
    labelpad=5,
    simmetric=True,
):

    abs_lim = max(abs(vmin), abs(vmax))
    if simmetric:
        real_max = abs_lim
        real_min = -abs_lim
    else:
        real_max = vmax
        real_min = vmin

    fig, ax = plot_cbar(
        cmap=cmap,
        title=title,
        vmin=real_min,
        vmax=real_max,
        locator=locator,
        labelpad=labelpad,
        horizontal=False,
        percent=False,
        rotation=90,
        figsize=figsize,
    )
    ax.set_ylim(vmin, vmax)

    filename_v = filename.parent / (filename.stem + "_v" + filename.suffix)
    fig.savefig(filename_v, bbox_inches="tight", transparent=True)
    plt.close(fig)

    fig, ax = plot_cbar(
        cmap=cmap,
        title=title,
        vmin=real_min,
        vmax=real_max,
        locator=locator,
        labelpad=labelpad,
        horizontal=True,
        percent=False,
        rotation=0,
        figsize=figsize[::-1],
    )
    ax.set_xlim(vmin, vmax)
    fig.savefig(filename, bbox_inches="tight", transparent=True)



def scatter_plots(
    df,
    hue,
    x="cca_1",
    y="cca_2",
    rois=["visual_cortex", "ventral_hub", "dorsal_hub"],
    s=5,
    axis_names=["CCA 1", "CCA 2"],
    palettes={},
    hue_orders={},
    histograms=False,
    margin=0.2,
    transpose=False,
):
    if isinstance(hue, str):
        hue = [hue]

    n_rows = len(hue)
    n_columns = len(rois)

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(s * n_columns, s * n_rows))

    if transpose:
        axes = axes.T

    for i, roi in enumerate(rois):
        df_scatter = df.query(f"roi=='{roi}'")[[x, y] + hue].copy().sample(frac=1)
        
        # Normalize x, y
        df_scatter[x] -= df_scatter[x].min()
        df_scatter[x] /= df_scatter[x].max()
        df_scatter[y] -= df_scatter[y].min()
        df_scatter[y] /= df_scatter[y].max()

        for j, h in enumerate(hue):
            palette = palettes.get(h)
            hue_order = hue_orders.get(h)
            
            ax = axes[j, i]
            legend = i == 0 #n_columns - 1
            sns.scatterplot(data=df_scatter, x=x, y=y, hue=h, ax=ax, legend=legend, palette=palette, hue_order=hue_order)
            if legend:
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    ncol=1,
                    title=h.replace("_", " ").title(),
                )
            sns.despine(ax=ax)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(roi.replace("_", " ").title())
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(axis_names[1])
            ax.axis("equal")
            ax.set_xlim(-margin, 1 + margin)
            ax.set_ylim(-margin, 1 + margin)
            #Add an auxiliary histogram to the x axis with a distplots
            #Add the axis to the bottom
            if j > 0 and histograms:
                big_position = ax.get_position()
                # Small axis (x0, y0, x1, y1)
                small_position = [big_position.x0, big_position.y0, big_position.width, big_position.height/10]
                ax_hist = fig.add_axes(small_position, frame_on=True)
                sns.kdeplot(data=df_scatter, x=x, ax=ax_hist, color="gray", hue=h, fill=True, common_norm=False, legend=False, zorder=-1000, hue_order=hue_order, palette=palette)
                # Only let the x axis visible
                ax_hist.axis("off")
                # Reverse the y axis

    if transpose:
        axes = axes.T

    return fig, axes

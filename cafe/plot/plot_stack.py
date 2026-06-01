import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .._logging import logger
from ..data import FateAnnData


# TODO: add docs
def plot_stack(
    fadata: FateAnnData,
    pseudotime_key: str = None,
    model_name: str = None,
    cluster: str = None,
    n_bins: int = 100,
    ax: plt.Axes = None,
    legend_loc: str = "best",
    bbox_to_anchor: tuple = (1, 0.5),
    save: str | bool = None,
    return_proportions: bool = False,
):
    """
    Generate a stack plot showing the proportion of cell clusters over pseudotime.

    This function calculates a global pseudotime for each cell based on the trajectory,
    bins the data, and plots the density of each cluster over time as a stacked area chart.

    Args:
        fadata (FateAnnData): The annotated data object containing a trajectory.
        cluster (str): The key in `fadata.obs` for cluster annotations.
        n_bins (int, optional): Number of bins to discretize pseudotime into. Defaults to 100.
        ax (plt.Axes, optional): An existing matplotlib axes object to plot on. Defaults to None.
        legend_loc (str, optional): Location of the legend. Defaults to "center left".
        bbox_to_anchor (tuple, optional): Bounding box for the legend. Defaults to (1, 0.5).
    """
    if cluster is None:
        cluster = fadata.prior_information.get("cluster")
        logger.debug(f"extract '{cluster}' from prior infomation as parameter 'cluster' ")
    logger.debug(f"Generating stack plot for clusters in 'fadata.obs[{cluster}]'...")

    # --- 1. Get Trajectory Data and Calculate Global Pseudotime ---
    if pseudotime_key is None:
        pseudotime = fadata.get_trajectory_pseudotime(model_name=model_name)
    else:
        pseudotime = fadata.obs[pseudotime_key]

    # --- 2. Prepare Data for Plotting ---
    plot_df = pd.DataFrame({"pseudotime": pseudotime, "cluster": fadata.obs[cluster]})

    # 根据伪时间类型选择离散类别或连续分箱
    time_series = plot_df["pseudotime"]
    is_discrete = ptypes.is_categorical_dtype(time_series) or ptypes.is_object_dtype(time_series)

    if is_discrete:
        # 离散时间点：不分箱，直接按类别聚合
        if ptypes.is_categorical_dtype(time_series):
            time_order = time_series.cat.categories.tolist()
        else:
            # 对于 object/string，按出现顺序或自然排序
            time_order = pd.unique(time_series)

        density_df = plot_df.groupby(["pseudotime", "cluster"]).size().unstack(fill_value=0).reindex(time_order).fillna(0)
        density_proportions = density_df.div(density_df.sum(axis=1), axis=0)
        x_vals = np.arange(len(time_order))
        x_tick_labels = time_order
    else:
        # 连续时间：分箱为面积堆叠
        pseudotime_bins = np.linspace(time_series.min(), time_series.max(), n_bins)
        plot_df["time_bin"] = pd.cut(plot_df["pseudotime"], bins=pseudotime_bins, labels=pseudotime_bins[:-1], right=False)
        density_df = plot_df.groupby(["time_bin", "cluster"]).size().unstack(fill_value=0)
        density_proportions = density_df.div(density_df.sum(axis=1), axis=0)
        x_vals = density_proportions.index.astype(float)
        x_tick_labels = None

    # --- 3. Plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Get cluster colors from the anndata object
    cluster_names = density_proportions.columns.tolist()
    try:
        colors = [fadata.uns[f"{cluster}_colors"][fadata.obs[cluster].cat.categories.tolist().index(c)] for c in cluster_names]
    except (KeyError, ValueError):
        logger.warning(f"Could not find colors in 'fadata.uns[{cluster}_colors]'. Using default colormap.")
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_names)))

    # Use stackplot to create the stack graph
    ax.stackplot(
        x_vals,  # X-axis: time (bins or discrete indices)
        density_proportions.T,  # Y-axis: proportions for each cluster
        labels=cluster_names,
        colors=colors,
        alpha=0.8,
    )

    # --- 4. Formatting ---
    ax.set_xlabel("pseudotime")
    ax.set_yticks([])  # Hide y-axis ticks as they are not meaningful (proportions)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Add an arrow for the x-axis to indicate direction
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)

    ax.set_title("group")  # As in the example image
    if legend_loc is not None:
        ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, frameon=False)

    # 设置离散时间刻度标签（仅在离散模式）
    if x_tick_labels is not None:
        ax.set_xticks(np.arange(len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels, rotation=0)

    plt.tight_layout()

    if save is not None:
        if isinstance(save, bool) and save:
            save = f".cafe/{fadata.id}/img/stack{pseudotime_key if pseudotime_key else model_name}.png"
        plt.savefig(save, bbox_inches="tight")
        logger.debug(f"save trajectory plot to '{save}'")

    if return_proportions:
        return density_proportions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from .._logging import logger
from ..data import FateAnnData


def plot_pseudotime_embedding(fadata, model_name: str = None, basis: str = None, pseudotime_key: str = None):
    if basis is None:
        basis = fadata.prior_information.get("basis", "X_umap")

    if pseudotime_key is not None:
        # use provided pseudotime key
        pseudotime = fadata.obs[pseudotime_key]
    else:
        # pseudtime not provided, compute milestone network related pseudotime
        pseudotime = fadata.get_trajectory_pseudotime(model_name=model_name)
        pseudotime_key = "_trajectory_pseudotime"
        fadata.obs[pseudotime_key] = pseudotime

    sc.pl.embedding(fadata, color=pseudotime_key, basis=basis, cmap="viridis")


def plot_pseudotime_stack(
    fadata: FateAnnData,
    model_name: str = None,
    cluster: str = None,
    pseudotime_key: str = None,
    n_bins: int = 20,
    ax: plt.Axes = None,
    legend_loc: str = "center left",
    bbox_to_anchor: tuple = (1, 0.5),
):
    """Plot stack, showing the proportion of cell clusters over pseudotime.

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
    logger.debug(f"Generating stream plot for clusters in 'fadata.obs[{cluster}]'...")
    if model_name is None:
        model_name = fadata.model_name
    if cluster is None:
        cluster = fadata.prior_information.get("cluster")
        logger.debug(f"extract '{cluster}' from prior infomation as parameter 'cluster' ")

    # --- 1. get global pseudotime ---
    if pseudotime_key is not None:
        # use provided pseudotime key
        pseudotime = fadata.obs[pseudotime_key]
    else:
        # pseudtime not provided, compute milestone network related pseudotime
        pseudotime = fadata.get_trajectory_pseudotime(model_name=model_name)

    # --- 2. prepare data for plotting ---
    plot_df = pd.DataFrame({"pseudotime": pseudotime, "cluster": fadata.obs[cluster]})

    # discretize pseudotime into bins
    pseudotime_bins = np.linspace(plot_df["pseudotime"].min(), plot_df["pseudotime"].max(), n_bins)
    plot_df["time_bin"] = pd.cut(plot_df["pseudotime"], bins=pseudotime_bins, labels=pseudotime_bins[:-1], right=False)

    # pivot the table to get counts of each cluster in each time bin
    density_df = plot_df.groupby(["time_bin", "cluster"]).size().unstack(fill_value=0)

    # normalize to get proportions (so the total height is always 1)
    density_proportions = density_df.div(density_df.sum(axis=1), axis=0)

    # --- 3. plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # get cluster colors from the anndata object
    cluster_names = density_proportions.columns.tolist()
    try:
        colors = [fadata.uns[f"{cluster}_colors"][fadata.obs[cluster].cat.categories.tolist().index(c)] for c in cluster_names]
    except (KeyError, ValueError):
        logger.warning(f"Could not find colors in 'fadata.uns[{cluster}_colors]'. Using default colormap.")
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_names)))

    # use stackplot to create the stream graph
    ax.stackplot(
        density_proportions.index.astype(float),  # X-axis: time bins
        density_proportions.T,  # Y-axis: proportions for each cluster
        labels=cluster_names,
        colors=colors,
        alpha=0.8,
    )

    # --- 4. formatting ---
    ax.set_xlabel("pseudotime")
    ax.set_yticks([])  # Hide y-axis ticks as they are not meaningful (proportions)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # add an arrow for the x-axis to indicate direction
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)

    ax.set_title("group")  # As in the example image
    ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, frameon=False)

    plt.tight_layout()

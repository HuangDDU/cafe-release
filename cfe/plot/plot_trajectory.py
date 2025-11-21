from collections.abc import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import norm

from .._logging import logger
from ..data import FateAnnData
from .add_color import add_milestone_cell_color, add_milestone_color


def plot_trajectory(
    fadata: FateAnnData,
    model_name: str | Sequence[str] = None,
    color: str | Sequence[str] = None,
    basis: str = None,
    curve: bool = True,
    layout_by_row: str = "color",
    milestone_color: str | list = None,
    color_trajectory: str = None,
    size_milestones: int = 30,
    size_transitions: int = 2,
    waypoint_wrapper_kwargs: dict = {},
    recompute_trajectory_embedding: bool = False,
    save: bool | str = None,
    **sc_pl_embedding_kwargs,
):
    """Plot cell embedding and trajectory with different color for now model by fadata.model_name
    ref: pydynverse/plot/plot_dimred.plot_dimred

    Args:
        fadata (FateAnnData): FateAnnData object with trajectory.
        model_name (str | Sequence[str], optional): model name(s).
        color (str | Sequence[str], optional): Color(s), default extracted from prior information.
        basis (str, optional): embedding basis.
        curve (bool, optional): whether to plot a curve.
        layout_by_row (str, optional): layout by row.
        milestone_color (str | list, optional): milestone color(s) to use for plotting.
        color_trajectory (str, optional): trajectory color.
        size_milestones (int, optional): milestone point size.
        size_transitions (int, optional): waypoint on trajectory curve size.
        waypoint_wrapper_kwargs (dict, optional): additional keyword arguments for waypoint wrapper.
        recompute_trajectory_embedding (bool, optional): whether to recompute trajectory embedding.
        save (str, optional): Path to save the plot.
        sc_pl_embedding_kwargs (dict, optional): additional keyword arguments for scanpy embedding plot.
    Returns:
        axes
    """

    if model_name is None:
        model_name = fadata.model_name
    if color is None:
        color = fadata.prior_information.get("cluster")
        logger.debug(f"extract '{color}' from prior infomation as parameter 'color' ")
    if basis is None:
        basis = fadata.prior_information.get("basis")
        logger.debug(f"extract '{basis}' from prior infomation as parameter 'basis' ")

    model_name_list = [model_name] if isinstance(model_name, str) else model_name
    color_list = [color] if isinstance(color, str) else color

    if len(model_name_list) == 1:
        layout_by_row = "model"  # only one model as row
    if len(color_list) == 1:
        layout_by_row = "color"  # only one color as row

    # create subplots
    if layout_by_row == "model":
        row_list, col_list = model_name_list, color_list
    elif layout_by_row == "color":
        row_list, col_list = color_list, model_name_list
    n_rows = len(row_list)
    n_cols = len(col_list)
    figsize = sc_pl_embedding_kwargs.pop("figsize", (7 * n_cols, 5 * n_rows))  # replace sc plt figsize
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # multiple model and color support
    for i, model_name in enumerate(model_name_list):
        # trajectory embedding extraction or calculation
        trajectory_embedding = fadata.get_trajectory_embedding(basis, model_name)  # trajectory embedding for specific basis
        if recompute_trajectory_embedding or trajectory_embedding is None:
            logger.debug(f"calculate new trajectory embedding for model_name:{model_name}, basis:{basis}.")
            # new trajectory embedding, project and save
            # project waypoint to embedding space
            cell_positions = pd.DataFrame(data=fadata.obsm[basis][:, :2], columns=["comp_1", "comp_2"])
            cell_positions["cell_id"] = fadata.obs.index
            waypoint_projection = project_waypoints(fadata, cell_positions, waypoint_wrapper_kwargs, model_name)
            # plot waypoint to show trajectory
            wp_segments = waypoint_projection["segments"]  # projection to trajectory
            milestone_positions = wp_segments[wp_segments["milestone_id"].apply(lambda x: x is not None)]  # only save waypoint on milestone
            # save trajectory embedding which is related to cell embbeding
            fadata.set_trajectory_embedding(wp_segments, milestone_positions, basis, model_name)
        else:
            # old trajectory embedding, read from fadata
            milestone_positions = trajectory_embedding["milestone_positions"]
            wp_segments = trajectory_embedding["wp_segments"]
        # temporal dataframe csv file for cellxgene visualization
        # milestone_positions.to_csv(f"tmp_milestone_positions.csv")
        # wp_segments.to_csv(f"tmp_wp_segments.csv")
        # print("Successfully write 'tmp_milestone_positions.csv' and 'tmp_wp_segments.csv' for cellxgene visualization")

        milestone_wrapper = fadata.get_milestone_wrapper(model_name)
        for j, color in enumerate(color_list):
            if layout_by_row == "model":
                ax = axes[i, j]  # row is model_name, col is color
            else:
                ax = axes[j, i]  # row is color, col is model_name
            logger.debug(f"plot_trajectory for model_name:'{model_name}', color:'{color}'")
            if color == "milestone":
                # add milestone mixed color
                milestone_id_list = milestone_wrapper["id_list"]
                milestone_percentages = milestone_wrapper["milestone_percentages"]
                milestone_color_list = add_milestone_color(len(milestone_id_list))
                milestone_color_dict = dict(zip(milestone_id_list, milestone_color_list))
                milestone_wrapper.color_list = [mcolors.to_hex(mc) for mc in milestone_color_list]  # add color for CXG visualization
                fadata.uns["milestone_color_dict"] = milestone_color_dict
                cell_color_df = add_milestone_cell_color(milestone_color_dict, milestone_percentages)
                fadata.obs["milestone"] = pd.Categorical(fadata.obs.index, categories=fadata.obs.index.tolist())
                missing_cell_color = "#808080"
                if cell_color_df.shape[0] != fadata.n_obs:
                    logger.warning(f"milestone cell color length not equal to cell number! set missing color as '{missing_cell_color}'.")
                fadata.uns["milestone_colors"] = [cell_color_df.loc[i] if i in cell_color_df.index else missing_cell_color for i in fadata.obs.index]

            # base scanpy embedding scatter plot
            sc_pl_embedding_kwargs["title"] = f"{fadata.get_parsed_model_name(model_name)}({color})"  # add title for subplot
            sc.pl.embedding(fadata, color=color, basis=basis, ax=ax, show=False, **sc_pl_embedding_kwargs)  # base cell embedding

            # legend remove
            if color == "milestone" or (layout_by_row == "color" and i < len(model_name) - 1):
                # milestone colors are too many, remove legend to save space
                # when color is row, only show legend for the last model
                ax.legend().remove()  # remove legend for color with milestone, but it waste time for show and remove

            # milestone and waypoint trajectory plot
            # TODO: trajectory plot keep unchange in the color loop, but it should plot for every ax.
            directed = milestone_wrapper["milestone_network"]["directed"].any()
            if curve:
                # draw milestone
                ax.scatter(milestone_positions["comp_1"], milestone_positions["comp_2"], c="black", s=size_milestones)  # waypoint scatter

                # connect waypoint scatter points into a curve
                for g in wp_segments["group"].unique():
                    wp_segments_g = wp_segments[wp_segments["group"] == g]
                    ax.plot(wp_segments_g["comp_1"], wp_segments_g["comp_2"], c="black", linewidth=size_transitions)

                # plot arrow at the midpoint of edge.
                if directed:

                    def get_arrow_df(group):
                        group = group.sort_values(by="percentage")
                        start = group.iloc[0]
                        end = group.iloc[-1]
                        # scale vector to aimed norm length for smooth curve
                        dx = end["comp_1"] - start["comp_1"]
                        dy = end["comp_2"] - start["comp_2"]
                        target_norm = 0.01
                        scale = target_norm / np.linalg.norm([dx, dy])
                        dx = scale * dx
                        dy = scale * dy
                        s = pd.Series({"x": start["comp_1"], "y": start["comp_2"], "dx": dx, "dy": dy})
                        return s

                    arrow_df = wp_segments[wp_segments["arrow"]].groupby("group").apply(get_arrow_df)
                    ax.quiver(arrow_df["x"], arrow_df["y"], arrow_df["dx"], arrow_df["dy"])
                if color_trajectory is None:
                    # TODO: add color to trajectory
                    pass
                else:
                    pass
            else:
                # directly use NetworkX to draw milestone network
                G = nx.from_pandas_edgelist(
                    milestone_wrapper["milestone_network"],
                    source="from",
                    target="to",
                    create_using=nx.DiGraph if directed else nx.Graph,
                )

                # get milestone positions
                def get_milestone(row):
                    f, t = row["group"].split("---")
                    if row["percentage"] == 0:
                        return f
                    else:
                        return t

                milestone_positions.apply(lambda row: get_milestone, axis=1)
                milestone_positions["milestone_id"] = milestone_positions.apply(lambda row: get_milestone(row), axis=1)
                milestone_positions = milestone_positions.groupby("milestone_id").apply(lambda x: x.iloc[0]).reset_index(drop=True)
                pos = dict(zip(milestone_positions["milestone_id"], milestone_positions[["comp_1", "comp_2"]].values))
                milestone_color_dict = fadata.uns["milestone_color_dict"]

                nx.draw_networkx(
                    G=G,
                    pos=pos,
                    node_color=[milestone_color_dict[node] for node in G.nodes],
                    width=3,
                    arrowsize=15,
                    linewidths=3,
                    edgecolors="black",
                    ax=ax,
                )

                # TODO: legend  setting

    if save is not None:
        if isinstance(save, bool) and save:
            save = f".cfe/{fadata.id}/img/trajectory_{basis}_{'_'.join(model_name_list)}.png"
        plt.savefig(save)
        logger.debug(f"save trajectory plot to '{save}'")
    return axes


def project_waypoints(
    fadata: FateAnnData,
    cell_positions: pd.DataFrame,
    waypoint_wrapper_kwargs: dict = {},
    model_name: str = None,
    trajectory_projection_sd: float = None,
) -> dict:
    """projectory waypoint into embbeding space

    ref: pydynverse/plot/project_waypoints.project_waypoints_coloured

    Args:
        fadata (FateAnnData): FateAnnData object with trajectory.
        cell_positions (pd.DataFrame): cell embedding position.
        trajectory_projection_sd (float, optional): distance scale of waypoint projection.

    Returns:
        dict: waypoint_projection dict
    """
    # if waypoints is None:
    # select waypoint
    logger.debug("add waypoints")
    milestone_wrapper = fadata.get_milestone_wrapper(model_name)
    fadata.add_waypoints(milestone_wrapper, model_name, waypoint_wrapper_kwargs)
    waypoints = fadata.get_waypoint_wrapper(model_name)
    logger.debug(f"add waypoints shape is {waypoints['waypoint_geodesic_distances'].shape} for '{model_name}', finished!")

    if trajectory_projection_sd is None:
        trajectory_projection_sd = sum(milestone_wrapper["milestone_network"]["length"]) * 0.05

    wps = waypoints
    # wps["waypoint_network"] = wps["waypoint_network"].rename({"from_milestone_id": "milestone_id_from", "to_milestone_id": "milestone_id_to"})

    # calculate wayppoint embedding based geodesic distances and gaussian kernel
    # calculate weight
    weights = wps["waypoint_geodesic_distances"].values.astype(float)
    weights = np.nan_to_num(weights)
    weights = norm.pdf(weights, scale=trajectory_projection_sd)  # gaussian kernel, the longer the distance, the smaller the weight
    weights /= weights.sum(axis=1, keepdims=True)  # weight normalization
    # get cell embedding
    positions = cell_positions[["cell_id", "comp_1", "comp_2"]].set_index("cell_id")
    positions = positions.loc[wps["waypoint_geodesic_distances"].columns]
    # calcate waypoint embedding base on weight
    result = np.dot(weights, positions)
    result_df = pd.DataFrame(result, columns=["comp_1", "comp_2"])
    result_df["waypoint_id"] = wps["waypoint_geodesic_distances"].index
    # merge waypoint embedding
    waypoint_positions = pd.merge(result_df, wps["waypoints"], on="waypoint_id")

    # merge waypoint progressions
    segments = pd.merge(waypoint_positions, wps["waypoint_progressions"], on="waypoint_id")
    segments["group"] = segments.apply(lambda x: f"{x['from']}---{x['to']}", axis=1)

    def calculate_closest_and_arrow(group):
        # choose the middle waypoint of a milestone network edege, where the percentage is closest to 0.5
        closest_index = (group["percentage"] - 0.5).abs().idxmin()
        group["arrow"] = (group.index == closest_index) | (group.index == closest_index + 1)  # arrow column
        return group

    segments = segments.groupby("group").apply(calculate_closest_and_arrow).reset_index(drop=True)

    waypoint_projection = {"segments": segments}

    return waypoint_projection

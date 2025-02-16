from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc

from .._logging import logger
from ..data import FateAnnData
from .add_color import add_milestone_color, add_milestone_cell_color


def plot_trajectory(
        fadata: FateAnnData,
        color: str | list = "milestone",
        basis: str = "umap",
        size_milestones: int = 30,
        size_transitions: int = 2,
        color_trajectory: str = None,
        save: str = None,
        ** sc_pl_embedding_kwargs
) -> None:
    """Plot cell embedding and trajectory with different color for now model by fadata.model_name

     ref: pydynverse/plot/plot_dimred.plot_dimred

    Args:
        fadata (FateAnnData): FateAnnData object with trajectory.
        basis (str, optional): embedding name in .obsm. key.
        size_milestones (int, optional): milestone point size.
        size_transitions (int, optional): waypoint on trajectory curve size.
        color_trajectory (str, optional): trajectory color.
    """

    # TODO: a fdata, a method => a fdata, many methods
    logger.debug("plot_trajectory")
    if (color == "milestone") or ((isinstance(color, list)) and ("milestone" in color)):
        # add milestone mixed color
        milestone_id_list = fadata.milestone_wrapper.id_list
        milestone_percentages = fadata.milestone_wrapper.milestone_percentages
        milestone_color_list = add_milestone_color(len(milestone_id_list))
        milestone_color_dict = dict(zip(milestone_id_list, milestone_color_list))
        cell_color_df = add_milestone_cell_color(milestone_color_dict, milestone_percentages)
        fadata.obs["milestone"] = pd.Categorical(fadata.obs.index, categories=fadata.obs.index.tolist())
        fadata.uns["milestone_colors"] = cell_color_df.loc[fadata.obs.index].values

    # base embedding
    ax_list = sc.pl.embedding(fadata, color=color, basis=basis, **sc_pl_embedding_kwargs, show=False)

    # project waypoint to embedding space
    cell_positions = pd.DataFrame(data=fadata.obsm[f"X_{basis}"][:, :2], columns=["comp_1", "comp_2"])
    cell_positions["cell_id"] = fadata.obs.index
    waypoint_projection = project_waypoints(fadata, cell_positions)

    # plot waypoint to show trajectory
    wp_segments = waypoint_projection["segments"]  # projection to trajectory
    milestone_positions = wp_segments[wp_segments["milestone_id"].apply(lambda x: x is not None)]  # only save waypoint on mileston

    # plot waypoint curve
    ax_list = ax_list if isinstance(ax_list, list) else [ax_list]
    color = color if isinstance(color, list) else [color]
    for i in range(len(color)):
        ax = ax_list[i]
        c = color[i]
        if c == "milestone":
            ax.legend().remove() # remove legend for color with milestone , but it waste time for show and remove 

        ax.scatter(milestone_positions["comp_1"], milestone_positions["comp_2"], c="black", s=size_milestones)  # waypoint scatter
        # Connect waypoint scatter points into a curve
        for g in wp_segments["group"].unique():
            wp_segments_g = wp_segments[wp_segments["group"] == g]
            ax.plot(wp_segments_g["comp_1"], wp_segments_g["comp_2"], c="black", linewidth=size_transitions)

        # plot waypoint
        if fadata.milestone_wrapper["milestone_network"]["directed"].any():
            def get_arrow_df(group):
                group = group.sort_values(by="percentage")
                start = group.iloc[0]
                end = group.iloc[-1]
                s = pd.Series({
                    "x": start["comp_1"],
                    "y": start["comp_2"],
                    "dx": end["comp_1"] - start["comp_1"],
                    "dy": end["comp_2"] - start["comp_2"]}
                )
                return s
            arrow_df = wp_segments[wp_segments["arrow"]].groupby("group").apply(get_arrow_df)
            ax.quiver(arrow_df["x"], arrow_df["y"], arrow_df["dx"], arrow_df["dy"])
            if color_trajectory is None:
                # TODO: add color to trajectory
                pass
            else:
                pass
            
    if save is not None:
        plt.savefig(save)
    return ax


def project_waypoints(
    fadata: FateAnnData,
    cell_positions: pd.DataFrame,
    trajectory_projection_sd: float = None
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
    milestone_wrapper = fadata.milestone_wrapper
    fadata.add_waypoints(milestone_wrapper)
    waypoints = fadata.waypoint_wrapper
    logger.debug(f"add waypoints shape is {waypoints['waypoint_geodesic_distances'].shape}, finished!")

    if trajectory_projection_sd is None:
        trajectory_projection_sd = sum(milestone_wrapper["milestone_network"]["length"]) * 0.05

    wps = waypoints
    # wps["waypoint_network"] = wps["waypoint_network"].rename({"from_milestone_id": "milestone_id_from", "to_milestone_id": "milestone_id_to"})

    # calculate wayppoint embedding based geodesic distances and gaussian kernel
    # calculate weight
    weights = wps["waypoint_geodesic_distances"].values
    weights = norm.pdf(weights, scale=trajectory_projection_sd)  # gaussian kernel
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

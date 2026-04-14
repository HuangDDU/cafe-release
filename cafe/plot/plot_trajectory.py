from collections.abc import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm

from .._logging import logger
from ..data import FateAnnData


# TODO: add docs
def plot_trajectory(
    fadata: FateAnnData,
    model_name: str | Sequence[str] = None,
    color: str | Sequence[str] = None,
    basis: str = None,
    curve: bool = True,
    layout_by_row: str = "color",
    show_milestone_labels: bool = False,
    milestone_legend_loc: str = "on data",
    milestone_color: str | list = None,
    color_trajectory: str = "black",
    size_milestones: int = 30,
    size_transitions: int = 2,
    size_arrow: int = 10,
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
        show_milestone_labels (bool, optional): whether to show milestone labels.
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
    # setting default parameter for sc.pl.embedding
    sc_pl_embedding_kwargs.setdefault("frameon", False)

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
                missing_cell_color = "#808080"
                cell_color_dict = milestone_wrapper["cell_color_dict"]
                if len(cell_color_dict) != fadata.n_obs:
                    logger.warning(f"milestone cell color length not equal to cell number! set missing color as '{missing_cell_color}'.")
                fadata.obs["milestone"] = pd.Categorical(fadata.obs.index, categories=fadata.obs.index.tolist())
                fadata.uns["milestone_colors"] = [cell_color_dict[i] if i in cell_color_dict else missing_cell_color for i in fadata.obs.index]

            # base scanpy embedding scatter plot
            if "title" not in sc_pl_embedding_kwargs:
                sc_pl_embedding_kwargs["title"] = f"{fadata.get_parsed_model_name(model_name)}({color})"  # add title for subplot
            sc.pl.embedding(fadata, color=color, basis=basis, ax=ax, show=False, **sc_pl_embedding_kwargs)  # base cell embedding

            # legend remove
            if color == "milestone" or (layout_by_row == "color" and i < len(model_name_list) - 1):
                # milestone colors are too many, remove legend to save space
                # when color is row, only show legend for the last model
                ax.legend().remove()  # remove legend for color with milestone, but it waste time for show and remove

            # milestone and waypoint trajectory plot
            # TODO: trajectory plot keep unchange in the color loop, but it should plot for every ax.
            directed = milestone_wrapper["directed"]

            if show_milestone_labels or not curve:
                G = milestone_wrapper["milestone_network_G"]
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
                milestone_color_dict = milestone_wrapper["milestone_color_dict"]

            # plot trajectory
            if curve:
                # waypoint calculation and visulization
                # connect waypoint scatter points into a curve
                for g in wp_segments["group"].unique():
                    wp_segments_g = wp_segments[wp_segments["group"] == g]
                    ax.plot(wp_segments_g["comp_1"], wp_segments_g["comp_2"], c=color_trajectory, linewidth=size_transitions)

                if directed:
                    arrow_segments = wp_segments[wp_segments["arrow"]].copy()

                    head_length = 0.6 * size_arrow
                    head_width = 0.4 * size_arrow
                    dynamic_arrowstyle = f"-|>,head_length={head_length},head_width={head_width}"

                    for _, row in arrow_segments.iterrows():
                        # Find the next point on the same segment to define a short vector for the arrow direction
                        segment_group = wp_segments[wp_segments["group"] == row["group"]]
                        current_index = segment_group.index.get_loc(row.name)

                        if current_index < len(segment_group) - 1:
                            start_pos = (row["comp_1"], row["comp_2"])
                            end_pos = (segment_group.iloc[current_index + 1]["comp_1"], segment_group.iloc[current_index + 1]["comp_2"])

                            # Use FancyArrowPatch for better control: only draw the head
                            arrow = FancyArrowPatch(
                                posA=start_pos,
                                posB=end_pos,
                                arrowstyle=dynamic_arrowstyle,  # Style: only draw head at posB
                                connectionstyle="arc3,rad=0",  # Straight line
                                shrinkA=0,  # No gap at the start
                                shrinkB=1,  # No gap at the end
                                color=color_trajectory,
                                zorder=4,
                            )
                            ax.add_patch(arrow)
            else:
                # use network
                nx.draw_networkx_edges(
                    G=G,
                    pos=pos,
                    edge_color=color_trajectory,
                    width=3,
                    arrowsize=15,
                    ax=ax,
                )

            # plot milestone
            if show_milestone_labels:
                # use networkx
                nx.draw_networkx_nodes(
                    G=G,
                    pos=pos,
                    node_color=[milestone_color_dict[node] for node in G.nodes],
                    edgecolors="black",
                    ax=ax,
                )
                # Show milestone legend
                if milestone_legend_loc == "on data":
                    nx.draw_networkx_labels(G=G, pos=pos)
                else:
                    # usually right margin
                    from matplotlib.lines import Line2D

                    milestone_handles = [
                        Line2D([0], [0], marker="o", color="w", label=m_id, markerfacecolor=m_color, markersize=10, markeredgecolor="black")
                        for m_id, m_color in milestone_color_dict.items()
                    ]

                    # check if there is an existing legend (Scanpy's), set title for Scanpy legend and keep it
                    scanpy_legend = ax.get_legend()
                    if scanpy_legend:
                        scanpy_legend.set_title("Cells")
                        ax.add_artist(scanpy_legend)
                        bbox_to_anchor = (1.3, 0.5)  # shift latter legene
                    else:
                        bbox_to_anchor = (1.0, 0.5)
                    # Add milestone legend below the existing one
                    ax.legend(handles=milestone_handles, title="Milestones", loc="center left", bbox_to_anchor=bbox_to_anchor, frameon=False)

            else:
                ax.scatter(milestone_positions["comp_1"], milestone_positions["comp_2"], c="black", s=size_milestones)  # waypoint scatter

    if save is not None:
        if isinstance(save, bool) and save:
            save = f".cafe/{fadata.id}/img/trajectory_{basis}_{'_'.join(model_name_list)}.png"
        plt.savefig(save, bbox_inches="tight")
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
        if len(group) > 2:
            closest_index = (group["percentage"] - 0.5).abs().idxmin()
            group["arrow"] = group.index == closest_index
        else:
            group["arrow"] = False
        return group

    segments = segments.groupby("group").apply(calculate_closest_and_arrow).reset_index(drop=True)

    waypoint_projection = {"segments": segments}

    return waypoint_projection

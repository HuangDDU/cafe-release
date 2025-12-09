import itertools
from collections.abc import Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scanpy as sc

from .._logging import logger
from ..data import FateAnnData


# TODO: two layer loop for model_name and color can be optimized by copy plot among ax
def plot_graph(
    fadata: FateAnnData,
    model_name: str | Sequence[str] = None,
    color: str | Sequence[str] = None,
    layout_by_row: str = "color",
    nx_draw_kwargs: dict = {},
    recompute_milestone_embedding: bool = True,
    save: bool | str = None,
    **sc_pl_embedding_kwargs,
):
    """Plot DAG base on milestone network amd show cell embedding

    Args:
        fadata (FateAnnData): FateAnnData object with trajectory.
        model_name (str | Sequence[str], optional): model name(s).
        color (str | Sequence[str], optional): Color(s), default extracted from prior information.
        layout_by_row (str, optional): layout by row.
        nx_draw_kwargs (dict, optional): additional keyword arguments for networkx draw.
        sc_pl_embedding_kwargs (dict, optional): additional keyword arguments for scanpy embedding plot.
        recompute_milestone_embedding (bool, optional): whether to recompute milestone embedding.
        save (bool | str, optional): path to save the plot.
        sc_pl_embedding_kwargs (dict, optional): additional keyword arguments for scanpy embedding plot.

    Returns:
        axes: axes
    """
    if model_name is None:
        model_name = fadata.model_name
    if color is None:
        color = fadata.prior_information.get("cluster")

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
        milestone_wrapper = fadata.get_milestone_wrapper(model_name=model_name)  # extract milestone network
        milestone_id_list = milestone_wrapper.id_list
        milestone_network = milestone_wrapper.milestone_network
        milestone_percentages = milestone_wrapper.milestone_percentages
        divergence_regions = milestone_wrapper.divergence_regions
        is_directed = milestone_wrapper["directed"]
        milestone_embedding = None
        if recompute_milestone_embedding or milestone_embedding is None:
            logger.debug(f"calculate new milestone embedding for model_name:{model_name}.")
            G = nx.from_pandas_edgelist(
                milestone_network,
                source="from",
                target="to",
                edge_attr=True,
                create_using=nx.DiGraph if is_directed else nx.Graph,
            )
            for descrete_node in set(milestone_id_list) - set(G.nodes):
                # descrete node need external addition
                G.add_node(descrete_node)
            milestone_emb_dict = nx.nx_agraph.graphviz_layout(G, prog="dot")  # position
            # position fo cell
            milestone_emb_df = pd.DataFrame(milestone_emb_dict).T

            def mix_emb(mpg, emb_df=milestone_emb_df):
                # mix related milestone emb to get position for a cell
                mpg_emb = emb_df.loc[mpg["milestone_id"]]
                return mpg_emb.apply(lambda emb_dim: (emb_dim.array * mpg["percentage"].array)).sum()

            basis = "_milestone_network_emb"
            cell_emb_df = milestone_percentages.groupby("cell_id").apply(lambda mpg: mix_emb(mpg))
        else:
            # TODO: save in fadata
            # milestone_embedding = fadata.get_milestone_embedding(model_name=model_name)  # # TODO: save in fadata
            pass

        # may lose some cells in cell_emb_df
        if fadata.shape[0] != cell_emb_df.shape[0]:
            logger.warning(f"cell number mismatch when plot graph for model '{model_name}', skip this model.")
            fadata = fadata[cell_emb_df.index]
            fadata.obsm[basis] = cell_emb_df.values
        else:
            fadata.obsm[basis] = cell_emb_df.loc[fadata.obs.index].values

        for j, color in enumerate(color_list):
            if layout_by_row == "model":
                ax = axes[i, j]  # row is model_name, col is color
            else:
                ax = axes[j, i]  # row is color, col is model_name

            if color == "milestone":
                # color of cells
                cell_color_key = "milestone"
                missing_cell_color = "#808080"
                cell_color_dict = milestone_wrapper["cell_color_dict"]
                if len(cell_color_dict) != fadata.n_obs:
                    logger.warning(f"milestone cell color length not equal to cell number! set missing color as '{missing_cell_color}'.")
                fadata.obs[cell_color_key] = pd.Categorical(fadata.obs.index, categories=fadata.obs.index.tolist())
                fadata.uns[f"{cell_color_key}_colors"] = [
                    cell_color_dict[i] if i in cell_color_dict else missing_cell_color for i in fadata.obs.index
                ]

            # base scanpy embedding scatter plot
            # plot single str for color parameter
            # zorder: 1: line, 2: cell(scanpy), 3: milestone
            sc_pl_embedding_kwargs["title"] = f"{fadata.get_parsed_model_name(model_name)}({color})"  # add title for subplot
            sc.pl.embedding(fadata, basis=basis, color=color, show=False, zorder=2, ax=ax, **sc_pl_embedding_kwargs)

            # legend remove
            if color == "milestone" or (layout_by_row == "color" and i < len(model_name) - 1):
                # remove legend for color with milestone, but it waste time for show and remove
                ax.legend().remove()

            # TODO: nx plot keep unchange in the color loop, but it should plot for every ax.
            milestone_color_dict = milestone_wrapper["milestone_color_dict"]
            nx.draw(
                G,
                milestone_emb_dict,
                with_labels=True,
                node_color=[milestone_color_dict[node] for node in G.nodes],
                width=5,
                edge_color="gray",
                arrowstyle="simple",
                arrowsize=30,
                ax=ax,
                **nx_draw_kwargs,
            )
            if divergence_regions.shape[0] > 0:
                plot_divergence_region(divergence_regions, milestone_emb_dict, ax=ax)  # divergence regoin

            del fadata.obsm[basis]

    if save is not None:
        if isinstance(save, bool) and save:
            save = f".cafe/{fadata.id}/img/graph_{basis}_{'_'.join(model_name_list)}.png"
        plt.savefig(save)
        logger.debug(f"save trajectory plot to '{save}'")
    return axes


def plot_divergence_region(divergence_regions, milestone_emb_dict, ax):
    triangles = []
    for did in divergence_regions["divergence_id"].unique():
        rel_did = divergence_regions[divergence_regions["divergence_id"] == did]
        fr = rel_did[rel_did["is_start"]]["milestone_id"].tolist()[0]  # only one start
        tos = rel_did[~rel_did["is_start"]]["milestone_id"].tolist()
        de_df = pd.DataFrame(itertools.product(tos, tos), columns=["node1", "node2"])
        de_df = de_df[de_df["node1"] < de_df["node2"]]
        de_df["divergence_id"] = did
        de_df["start"] = fr
        triangles.append(de_df)
    triangles = pd.concat(triangles)
    triangles = triangles[["divergence_id", "start", "node1", "node2"]]

    # calc position
    milestone_positions = pd.DataFrame(milestone_emb_dict).T
    if (divergence_regions is not None) and (divergence_regions.shape[0] > 0):
        # divergece end edge
        divergence_edge_positions = triangles.rename(columns={"node1": "from", "node2": "to"})
        divergence_edge_positions[["comp_1_from", "comp_2_from"]] = divergence_edge_positions["from"].apply(lambda x: milestone_positions.loc[x])
        divergence_edge_positions[["comp_1_to", "comp_2_to"]] = divergence_edge_positions["to"].apply(lambda x: milestone_positions.loc[x])
        # divergence polygon area
        divergence_polygon_positions = triangles.copy()
        divergence_polygon_positions["triangle_id"] = [f"triangle_{i}" for i in range(triangles.shape[0])]
        divergence_polygon_positions = divergence_polygon_positions.melt(
            id_vars=["triangle_id"],
            value_vars=["start", "node1", "node2"],
            var_name="triangle_part",
            value_name="milestone_id",
        )
        divergence_polygon_positions[["comp_1", "comp_2"]] = divergence_polygon_positions["milestone_id"].apply(lambda x: milestone_positions.loc[x])
    else:
        divergence_edge_positions = pd.DataFrame(
            columns=["divergence_id", "start", "from", "to", "comp_1_from", "comp_2_from", "comp_1_to", "comp_2_to"]
        )
        divergence_polygon_positions = pd.DataFrame(columns=["triangle_id", "triangle_part", "milestone_id", "comp_1", "comp_2"])

    # plot
    dep = divergence_edge_positions
    x_edges = dep[["comp_1_from", "comp_1_to"]].T.values  # 2*n
    y_edges = dep[["comp_2_from", "comp_2_to"]].T.values  # 2*n
    ax.plot(x_edges, y_edges, color="lightgrey", linestyle="--", linewidth=5, zorder=1)
    dpp = divergence_polygon_positions
    for triangle_id in dpp["triangle_id"].unique():
        polygon_vertices = dpp[dpp["triangle_id"] == triangle_id][["comp_1", "comp_2"]].values  # extract polygon point
        polygon = patches.Polygon(polygon_vertices, closed=True, fill=True, color="lightgrey", alpha=0.5)
        ax.add_patch(polygon)

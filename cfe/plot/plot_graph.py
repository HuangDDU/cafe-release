import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import networkx as nx
import scanpy as sc

from ..data import FateAnnData
from .add_color import add_milestone_color, add_milestone_cell_color


def plot_graph(
    fadata: FateAnnData,
    color: str | list = "milestone",
    nx_draw_kwrags={},
    sc_pl_embedding_kwargs={},
):
    """Plot DAG base on milestone network

    Args:
        fadata (FateAnnData): FateAnnData
        save (str, optional): img path.

    Returns:
        _type_: _description_
    """

    # extract milestone network
    milestone_wrapper = fadata.milestone_wrapper
    milestone_id_list = milestone_wrapper.id_list
    milestone_network = milestone_wrapper.milestone_network
    milestone_percentages = milestone_wrapper.milestone_percentages
    divergence_regions = milestone_wrapper.divergence_regions
    is_directed = milestone_network["directed"].any()

    # color and position fo milestone
    milestone_color_list = add_milestone_color(len(milestone_id_list))  # color rgb
    milestone_color_dict = dict(zip(milestone_id_list, milestone_color_list))
    G = nx.from_pandas_edgelist(
        milestone_network,
        source="from",
        target="to",
        edge_attr=True,
        create_using=nx.DiGraph if is_directed else nx.Graph
    )
    milestone_emb_dict = nx.nx_agraph.graphviz_layout(G, prog="dot")  # position

    # position fo cell
    milestone_emb_df = pd.DataFrame(milestone_emb_dict).T

    def mix_emb(mpg):
        # mix related milestone emb to get position for a cell
        mpg_emb = milestone_emb_df.loc[mpg["milestone_id"]]
        return mpg_emb.apply(lambda emb_dim: (emb_dim.array * mpg["percentage"].array)).sum()
    basis = "milestone_network_emb"
    cell_emb_df = milestone_percentages.groupby("cell_id").apply(lambda mpg: mix_emb(mpg))
    fadata.obsm[basis] = cell_emb_df.loc[fadata.obs.index].values

    if "milestone" in color:
        # color of cells
        cell_color_key = "milestone"
        cell_color_df = add_milestone_cell_color(milestone_color_dict, milestone_percentages)
        fadata.obs[cell_color_key] = pd.Categorical(fadata.obs.index, categories=fadata.obs.index.tolist())
        fadata.uns[f"{cell_color_key}_colors"] = cell_color_df.loc[fadata.obs.index].values

    # plot
    # zorder: 1: line, 2: cell, 3: milestone
    ax_list = sc.pl.embedding(
        fadata,
        basis=basis,
        color=color,
        show=False,
        zorder=2,
        **sc_pl_embedding_kwargs
    )  # first plot embedding to get ax_list
    ax_list = ax_list if isinstance(ax_list, list) else [ax_list]
    color = color if isinstance(color, list) else [color]
    for i in range(len(color)):
        ax = ax_list[i]
        c = color[i]
        if c == "milestone":
            ax.legend().remove() # remove legend for color with milestone , but it waste time for show and remove 

        nx.draw(G,
                milestone_emb_dict,
                with_labels=True,
                node_color=[milestone_color_dict[node] for node in G.nodes],
                width=5,  # TODO: adjusted by cell size
                edge_color="gray",
                arrowstyle="simple",
                arrowsize=30,   # TODO: adjusted by cell size
                ax=ax,
                **nx_draw_kwrags,
                )
        if divergence_regions.shape[0] > 0:
            plot_divergence_region(divergence_regions, milestone_emb_dict, ax=ax)  # divergence regoin

        

        # tmp_kwargs = sc_pl_embedding_kwargs.copy()
        # tmp_kwargs["legend_loc"] = None if c == "milestone" else tmp_kwargs.get("legend_loc", None)
        # sc.pl.embedding(
        #     fadata,
        #     basis=basis,
        #     color=c,
        #     ax=ax,
        #     legend_loc=None,
        #     # **tmp_kwargs
        # )  # second plot embedding to put embbeding on top graph layer


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
            value_name="milestone_id"
        )
        divergence_polygon_positions[["comp_1", "comp_2"]] = divergence_polygon_positions["milestone_id"].apply(lambda x: milestone_positions.loc[x])
    else:
        divergence_edge_positions = pd.DataFrame(columns=["divergence_id", "start", "from", "to", "comp_1_from", "comp_2_from", "comp_1_to", "comp_2_to"])
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

    return ax

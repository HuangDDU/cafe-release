import networkx as nx

from ..._logging import logger

# from ..plot_pie import plot_pie
from ..plot_trajectory import plot_trajectory

DEFAULT_MODE = "embedding"


def plot_embedding(fadata, model_name: str = None, basis=None):
    raw_wrapper_dict = fadata.get_raw_wrapper_dict(model_name=model_name)
    cluster = raw_wrapper_dict["cluster"]

    # may lose some cells and milestone, like stavia pruning option
    mw = fadata.get_milestone_wrapper(model_name=model_name)
    detected_milestone_id_set = set(cluster.unique())
    milestone_id_set = set(mw.id_list)

    if not (detected_milestone_id_set == milestone_id_set):
        logger.warning(
            "detected milestone id set is different from milestone wrapper milestone id set, may lose some cells or milestone after pruning."
        )
        cluster = cluster[cluster.isin(milestone_id_set)]

    # plot trajectory
    axes = plot_trajectory(
        fadata,
        model_name=model_name,
        basis=basis,
        curve=False,
        show_milestone_labels=True,
    )
    ax = axes.flatten()[0]

    # connect cell and milestone by cluster
    # create edge dataframe
    edge_df = cluster.reset_index()
    edge_df.columns = ["source", "target"]
    G = nx.from_pandas_edgelist(edge_df, create_using=nx.Graph)
    # add node position
    pos = {}
    cell_id_list = cluster.index.tolist()
    pos.update(dict(zip(cell_id_list, fadata[cell_id_list].obsm["X_umap"].copy().tolist())))  # add cell node pos
    trajectory_embedding = fadata.get_trajectory_embedding(model_name=model_name, basis=basis)
    milestone_positions = (
        trajectory_embedding["milestone_positions"].groupby("milestone_id").first()[["comp_1", "comp_2"]]
    )  # filter unique milestone pos
    pos.update(milestone_positions.apply(list, axis=1).to_dict())  # add milestone node pos
    # plot edges
    nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.5, edge_color="gray", width=0.5)

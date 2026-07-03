import networkx as nx
import pandas as pd
import scanpy as sc

DEFAULT_MODE = "embedding"


def plot_embedding(fadata, model_name: str = None, cluster: str = None, basis: str = None):
    if cluster is None:
        cluster = fadata.prior_information.get("cluster")
    if basis is None:
        basis = fadata.prior_information.get("basis")

    raw_wrapper_dict = fadata.get_raw_wrapper_dict(model_name)
    mw = fadata.get_milestone_wrapper(model_name)

    cell_graph = raw_wrapper_dict["cell_graph"]
    milestone_network = mw["milestone_network"].copy()
    milestone_network["from"] = milestone_network["from"].apply(lambda x: x.replace("milestone_", ""))
    milestone_network["to"] = milestone_network["to"].apply(lambda x: x.replace("milestone_", ""))
    all_cell_id_list = list(pd.unique(pd.concat([cell_graph["from"], cell_graph["to"]])))
    filtered_cell_id_list = list(pd.unique(pd.concat([milestone_network["from"], milestone_network["to"]])))

    # base scanpy plot
    ax = sc.pl.embedding(fadata, color=cluster, basis=basis, frameon=False, show=False)

    # plot all cells in graph
    G_all = nx.from_pandas_edgelist(cell_graph, source="from", target="to", create_using=nx.Graph)
    pos_all = dict(zip(all_cell_id_list, fadata[all_cell_id_list].obsm[basis].copy().tolist()))  # add cell node pos
    nx.draw_networkx_edges(G_all, pos=pos_all, ax=ax, alpha=0.5, edge_color="gray", width=0.5)

    # plot simplified cells
    G_filtered = nx.from_pandas_edgelist(milestone_network, source="from", target="to", create_using=nx.Graph)
    pos_filtered = dict(zip(filtered_cell_id_list, fadata[filtered_cell_id_list].obsm[basis].copy().tolist()))  # add cell node pos
    nx.draw_networkx_edges(G_filtered, pos=pos_filtered, ax=ax, alpha=1, edge_color="black", width=1)

    return ax

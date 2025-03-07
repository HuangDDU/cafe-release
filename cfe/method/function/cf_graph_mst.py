import pandas as pd
import networkx as nx
import anndata as ad
import scanpy as sc


def cf_graph_mst(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # 1. prepare data
    adata = adata.copy()
    cell_ids = adata.obs.index

    # 2. preprocess
    sc.pp.pca(adata, n_comps=parameters["ndim"])

    # 3.execute method
    # construct the minimum spanning tree between cells directly
    distance_metric = parameters["distance_metric"]
    sc.pp.neighbors(adata, metric=distance_metric)
    G = nx.from_scipy_sparse_array(adata.obsp["distances"])  # construct graph from a sparse matrix
    cell_mst = nx.minimum_spanning_tree(G, weight="weight")

    # 4. extract results
    cell_graph = nx.to_pandas_edgelist(cell_mst, source="from", target="to").rename(columns={"weight": "length"})
    cell_graph["from"] = cell_graph["from"].apply(lambda x: cell_ids[x])
    cell_graph["to"] = cell_graph["to"].apply(lambda x: cell_ids[x])
    # to_keep = pd.Series(data=True, index=cell_ids)

    # 5. save results
    trajectory_dict = {
        "cell_graph": cell_graph,
        "to_keep": None,  # keep all
    }
    return trajectory_dict

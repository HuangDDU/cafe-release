import anndata as ad
import networkx as nx

# import pandas as pd
import scanpy as sc


def graph_mst(
    adata: ad.AnnData,
    repreprocess: bool = True,
    pca_ndim: int = 5,
    neighbors_kwargs: dict = {},
):
    # 1. preprocess
    if repreprocess:
        sc.pp.pca(adata, n_comps=pca_ndim)

    # 2.execute method
    sc.pp.neighbors(adata, **neighbors_kwargs)  # recompute neighbors
    cell_ids = adata.obs.index
    G = nx.from_scipy_sparse_array(adata.obsp["distances"])  # construct graph from a sparse matrix
    cell_mst = nx.minimum_spanning_tree(G, weight="weight")  # construct the minimum spanning

    # 3. extract results
    cell_graph = nx.to_pandas_edgelist(cell_mst, source="from", target="to").rename(columns={"weight": "length"})
    cell_graph["from"] = cell_graph["from"].apply(lambda x: cell_ids[x])
    cell_graph["to"] = cell_graph["to"].apply(lambda x: cell_ids[x])
    # to_keep = pd.Series(data=True, index=cell_ids)

    # 4. save results
    trajectory_dict = {
        "cell_graph": cell_graph,
        "to_keep": None,  # keep all
    }
    return trajectory_dict


def cf_graph_mst(
    adata: ad.AnnData,
    prior_information: dict = None,
    parameters: dict = None,
    **kwargs,
):
    if (prior_information is None) and (parameters is None):
        # for new backend call, function(**kwargs)
        return graph_mst(adata, **kwargs)
    else:
        # for old backend call, function(prior_information, parameters)
        parameters.update(prior_information)
        return graph_mst(adata, **parameters)

#!/usr/local/bin/python3
from typing import Literal, Optional

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.pairwise import pairwise_distances


def projection_mst(
    adata: ad.AnnData,
    repreprocess: bool = True,
    pca_ndim: int = 5,
    basis: str = "X_pca",
    recluster: bool = True,
    cluster_key: str = "clusters",
    distance_metric: Optional[Literal["euclidean", "cosine", "manhattan", "cityblock", "l1", "l2"]] = "euclidean",
):
    # 1. preprocess
    adata.obs.reset_index(drop=True, inplace=True)
    if repreprocess and (basis == "X_pca"):
        sc.pp.pca(adata, n_comps=pca_ndim)
    X_emb = adata.obsm[basis]

    # 2. execute method
    # (1) if recluster cells, with the center point as a milestone
    if recluster:
        # new cluster
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)
        cluster_key = "leiden"
    # (2) Calculate the low dimensional coordinates of the clustering centers
    centers = np.array(list(adata.obs.groupby(cluster_key).apply(lambda x: X_emb[list(x.index)].mean(axis=0))))
    milestone_ids = [f"M{i}" for i in range(centers.shape[0])]
    centers = pd.DataFrame(centers, index=milestone_ids)
    # (3) Calculate the distance between cluster centers
    dis = pd.DataFrame(pairwise_distances(centers, metric=distance_metric), index=milestone_ids, columns=milestone_ids)
    disdf = pd.DataFrame(data=dis.unstack().reset_index().values, columns=["from", "to", "weight"])  # width data to long data
    # (4) Calculate the distance between milestones and construct the minimum spanning tree as the milestone network
    G = nx.from_pandas_edgelist(disdf, source="from", target="to", edge_attr="weight")
    mst = nx.minimum_spanning_tree(G, weight="weight")
    milestone_network = nx.to_pandas_edgelist(mst)
    milestone_network.rename(columns={"source": "from", "target": "to", "weight": "length"}, inplace=True)
    milestone_network["directed"] = False

    # 3. extract results
    comp_ids = [f"comp_{i+1}" for i in range(centers.shape[1])]
    X_emb = pd.DataFrame(X_emb, index=adata.obs.index, columns=comp_ids)
    milestone_emb = centers
    milestone_emb.columns = comp_ids

    # 4. save results
    trajectory_dict = {
        "milestone_network": milestone_network,
        "X_emb": X_emb,
        "milestone_emb": milestone_emb,
    }
    return trajectory_dict


def cf_projection_mst(
    adata: ad.AnnData,
    prior_information: dict = None,
    parameters: dict = None,
    **kwargs,
):
    if (prior_information is None) and (parameters is None):
        # for new backend call, function(**kwargs)
        return projection_mst(adata, **kwargs)
    else:
        # for old backend call, function(prior_information, parameters)
        parameters.update(prior_information)
        return projection_mst(adata, **parameters)

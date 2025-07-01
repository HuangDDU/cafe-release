#!/usr/local/bin/python3
import pickle

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.pairwise import pairwise_distances


def cf_projection_mst(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # 1. prepare data
    adata = adata.copy()
    adata.obs.reset_index(drop=True, inplace=True)

    # 2. preprocess
    sc.pp.pca(adata, n_comps=parameters["ndim"])
    X_emb = adata.obsm["X_pca"]

    # 3. execute method
    # (1) Cluster cells, with the center point as a milestone
    if "groups_id" not in prior_information:
        # new cluster
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)
        cluster_key = "leiden"
    else:
        # cluster in prior_information
        cluster_key = "mst_cluster"
        adata.obs[cluster_key] = prior_information["groups_id"]
    # (2) Calculate the low dimensional coordinates of the clustering centers
    centers = np.array(list(adata.obs.groupby(cluster_key).apply(lambda x: X_emb[list(x.index)].mean(axis=0))))
    milestone_ids = [f"M{i}" for i in range(centers.shape[0])]
    centers = pd.DataFrame(centers, index=milestone_ids)
    # (3) Calculate the distance between cluster centers
    distance_metric = parameters["distance_metric"]
    dis = pd.DataFrame(pairwise_distances(centers, metric=distance_metric), index=milestone_ids, columns=milestone_ids)
    disdf = pd.DataFrame(data=dis.unstack().reset_index().values, columns=["from", "to", "weight"])  # 转化为长数据

    # (4) Calculate the distance between milestones and construct the minimum spanning tree as the milestone network
    G = nx.from_pandas_edgelist(disdf, source="from", target="to", edge_attr="weight")
    mst = nx.minimum_spanning_tree(G, weight="weight")
    milestone_network = nx.to_pandas_edgelist(mst)
    milestone_network.rename(columns={"source": "from", "target": "to", "weight": "length"}, inplace=True)
    milestone_network["directed"] = False

    # 4. extract results
    comp_ids = [f"comp_{i+1}" for i in range(centers.shape[1])]
    X_emb = pd.DataFrame(X_emb, index=adata.obs.index, columns=comp_ids)
    milestone_emb = centers
    milestone_emb.columns = comp_ids

    # 5. save results
    trajectory_dict = {
        "milestone_network": milestone_network,
        "X_emb": X_emb,
        "milestone_emb": milestone_emb,
    }
    return trajectory_dict


if __name__ == "__main__":
    from parse_args import parse_args

    adata, prior_information, parameters, output_filename = parse_args()

    trajectory_dict = cf_projection_mst(adata, prior_information, parameters)

    with open(output_filename, "wb") as f:
        pickle.dump(trajectory_dict, f)
    print("projection_mst Finish!")

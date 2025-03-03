#!/usr/local/bin/python3
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import anndata as ad
import scanpy as sc

from sklearn.metrics.pairwise import pairwise_distances


def cf_projection_mst(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # 1. 数据构造
    adata = adata.copy()
    adata.obs.reset_index(drop=True, inplace=True)

    # 2. 执行PCA
    sc.pp.pca(adata, n_comps=parameters["ndim"])
    X_emb = adata.obsm["X_pca"]
    # n_comps = parameters.get("n_comps", 10)
    # n_gene = expression.shape[1]
    # if n_gene < n_comps:
    #     # 如果基因数小于n_comps，不降维
    #     n_comps = n_gene
    # else:
    #     sc.pp.pca(adata, n_comps=parameters["ndim"])

    # 3. 聚类细胞，中心点作为里程碑
    # （1）聚类细胞
    if "groups_id" not in prior_information:
        # 为了方便，这里直接调用scanpy的聚类方法leiden
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)
        cluster_key = "leiden"
    else:
        # 使用先验知识中的里程碑
        cluster_key = "mst_cluster"
        adata.obs[cluster_key] = prior_information["groups_id"]
    # （2）计算聚类中心的低维坐标
    centers = np.array(list(adata.obs.groupby(cluster_key).apply(lambda x: X_emb[list(x.index)].mean(axis=0))))
    milestone_ids = [f"M{i}"for i in range(centers.shape[0])]
    centers = pd.DataFrame(centers, index=milestone_ids)
    # （3）计算聚类中心间的距离
    distance_metric = parameters["distance_metric"]
    dis = pd.DataFrame(pairwise_distances(centers, metric=distance_metric), index=milestone_ids, columns=milestone_ids)
    disdf = pd.DataFrame(data=dis.unstack().reset_index().values, columns=["from", "to", "weight"])  # 转化为长数据

    # 4. 里程碑之间计算距离并构建最小生成树作为里程碑网络
    G = nx.from_pandas_edgelist(disdf, source="from", target="to", edge_attr="weight")
    mst = nx.minimum_spanning_tree(G, weight="weight")
    milestone_network = nx.to_pandas_edgelist(mst)
    milestone_network.rename(columns={"source": "from", "target": "to", "weight": "length"}, inplace=True)
    milestone_network["directed"] = False

    # 5. 结果封装保存
    comp_ids = [f"comp_{i+1}"for i in range(centers.shape[1])]
    X_emb = pd.DataFrame(X_emb, index=adata.obs.index, columns=comp_ids)
    milestone_emb = centers
    milestone_emb.columns = comp_ids

    trajectory_dict = {
        "milestone_network": milestone_network,
        "X_emb": X_emb,
        "milestone_emb": milestone_emb,
    }
    return trajectory_dict


if __name__ == "__main__":
    from parse_args import parse_args

    adata, prior_information, parameters, output_filename = parse_args()

    trajectory_dict = cf_mst(adata, prior_information, parameters)

    with open(output_filename, "wb") as f:
        pickle.dump(trajectory_dict, f)
    print("PAGA Finish!")

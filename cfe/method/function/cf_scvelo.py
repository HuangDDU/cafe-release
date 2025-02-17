import numpy as np
import pandas as pd

import anndata as ad
import scvelo as scv


def cf_scvelo(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # ref: https://scvelo.readthedocs.io/en/stable/VelocityBasics.html
    cluster_key = prior_information.get("cluster_key", "clusters")

    # 1. 数据构造
    adata = adata.copy()

    # 2. 预处理
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # 3. 方法调用
    scv.tl.velocity(adata)  # 高维速率计算
    scv.tl.velocity_graph(adata)  # 转移概率计算
    # scv.pl.velocity_embedding_stream(adata, basis="umap", show=False)  # 降维速率图展示

    # 4. PAGA计算milestone network有向图
    milestone_id_list = list(adata.obs[cluster_key].cat.categories)

    adata.uns["neighbors"]["distances"] = adata.obsp["distances"]
    adata.uns["neighbors"]["connectivities"] = adata.obsp["connectivities"]
    scv.tl.paga(adata, groups=cluster_key)
    df = scv.get_df(adata, "paga/transitions_confidence", precision=2).T
    df.index = milestone_id_list
    df.columns = milestone_id_list

    milestone_network = df.reset_index()\
        .rename(columns={'index': 'from'})\
        .melt(id_vars="from", var_name="to", value_name="length")\
        .query("`length` > 0")
    milestone_network["length"] = 1  # 暂时统一设置为1
    milestone_network["directed"] = True

    obs = adata.obs.reset_index()  # change index
    X_emb = adata.obsm["X_umap"]
    milestone_emb = np.array(list(obs.groupby(cluster_key).apply(lambda x: X_emb[list(x.index)].mean(axis=0))))
    milestone_emb = pd.DataFrame(milestone_emb, index=milestone_id_list)

    # 5. 结果封装保存
    trajectory_dict = trajectory_dict = {
        "milestone_network": milestone_network,
        "X_emb": X_emb,
        "milestone_emb": milestone_emb,
    }

    return trajectory_dict

import anndata as ad
import scvelo as scv


def cf_scvelo(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # ref: https://scvelo.readthedocs.io/en/stable/VelocityBasics.html

    # 1. 数据构造
    adata = adata.copy()

    # 2. 预处理
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # 3. 方法调用
    scv.tl.velocity(adata)  # 高维速率计算
    scv.tl.velocity_graph(adata)  # 转移概率计算
    scv.pl.velocity_embedding_stream(adata, basis="umap", show=False)  # 降维速率图展示

    # 4. 结果封装保存
    neighbors = adata.uns["neighbors"]
    neighbors['distances'] = adata.obsp["distances"]
    neighbors['connectivities'] = adata.obsp["connectivities"]
    
    trajectory_dict = {
        "neighbors": neighbors,
        "velocity": adata.layers["velocity"],
        "velocity_graph": adata.uns["velocity_graph"]
    }

    return trajectory_dict

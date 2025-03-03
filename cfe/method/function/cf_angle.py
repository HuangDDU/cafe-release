import numpy as np
import anndata as ad
import scanpy as sc


def cf_angle(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # 1. 数据构造
    adata = adata.copy()
    cell_ids = adata.obs.index

    # 2. 执行PCA
    sc.pp.pca(adata, n_comps=parameters["ndim"])

    # PCA线性空间->arctan2映射到环形空间
    X_pca = adata.obsm["X_pca"]
    pseudotime = np.arctan2(X_pca[:, 1], X_pca[:, 0]) / (2*np.pi) + 0.5

    trajectory_dict = {
        "pseudotime": pseudotime,
        "cycle": True,  # TODO：是否环形，需要调整
    }

    return trajectory_dict

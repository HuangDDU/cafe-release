import numpy as np
import anndata as ad
import scanpy as sc


def cf_angle(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # 1. prepare data
    adata = adata.copy()
    cell_ids = adata.obs.index

    # 2. preprocess
    sc.pp.pca(adata, n_comps=parameters["ndim"])

    # 3. execute method
    # PCA linear space -> arctan2 nonlinear space
    X_pca = adata.obsm["X_pca"]
    pseudotime = np.arctan2(X_pca[:, 1], X_pca[:, 0]) / (2*np.pi) + 0.5

    # 4. save results
    trajectory_dict = {
        "pseudotime": pseudotime,
    }

    return trajectory_dict

import anndata as ad
import numpy as np
import scanpy as sc


def cf_angle(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # 1. extract prior information and parameters
    repreprocess = parameters["repreprocess"]
    pca_ndim = parameters["pca_ndim"]
    basis = parameters["basis"]

    # 2. preprocess
    if repreprocess:
        sc.pp.pca(adata, n_comps=pca_ndim)

    # 3. execute method
    # embedding space -> arctan2 nonlinear space, extract first two components for arctan2
    X_emb = adata.obsm[basis]
    pseudotime = np.arctan2(X_emb[:, 1], X_emb[:, 0]) / (2 * np.pi) + 0.5

    # 4,5. extract and save results
    trajectory_dict = {
        "pseudotime": pseudotime,
    }

    return trajectory_dict

import anndata as ad
import numpy as np
import scanpy as sc


def angle(adata: ad.AnnData, repreprocess: bool = True, pca_ndim: int = 5, basis: str = "X_pca"):
    # 1. preprocess
    if repreprocess:
        sc.pp.pca(adata, n_comps=pca_ndim)

    # 2. execute method
    # embedding space -> arctan2 nonlinear space, extract first two components for arctan2
    X_emb = adata.obsm[basis]
    pseudotime = np.arctan2(X_emb[:, 1], X_emb[:, 0]) / (2 * np.pi) + 0.5

    # 3,4. extract and save results
    trajectory_dict = {
        "pseudotime": pseudotime,
    }

    return trajectory_dict


def cf_angle(
    adata: ad.AnnData,
    prior_information: dict = None,
    parameters: dict = None,
    **kwargs,
):
    if (prior_information is None) and (parameters is None):
        # for new backend call, function(**kwargs)
        return angle(adata, **kwargs)
    else:
        # for old backend call, function(prior_information, parameters)
        parameters.update(prior_information)
        return angle(adata, **parameters)

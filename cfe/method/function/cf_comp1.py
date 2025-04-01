import anndata as ad
import scanpy as sc


def cf_comp1(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # 1. prepare data
    adata = adata.copy()

    # 2. preprocess and execute method simutaneously with pca
    sc.pp.pca(adata, n_comps=parameters["ndim"])

    # 3. extract results
    pseudotime = adata.obsm["X_pca"][:, parameters["component"] - 1]

    # 4. save results
    trajectory_dict = {
        "pseudotime": pseudotime,
    }

    return trajectory_dict

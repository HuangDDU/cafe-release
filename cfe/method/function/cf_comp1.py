import anndata as ad
import scanpy as sc


def cf_comp1(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # 1. extract prior information and parametersQ
    repreprocess = parameters["repreprocess"]
    pca_ndim = parameters["pca_ndim"]
    basis = parameters["basis"]
    component = parameters["component"]

    # 2, 3. preprocess and execute method simutaneously with pca
    if repreprocess and (basis == "X_pca"):
        sc.pp.pca(adata, n_comps=pca_ndim)

    # 4. extract results
    pseudotime = adata.obsm[basis][:, component - 1]

    # 5. save results
    trajectory_dict = {
        "pseudotime": pseudotime,
    }

    return trajectory_dict

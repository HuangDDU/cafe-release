import numpy as np
import anndata as ad
import scanpy as sc
import scanpy.external as sce


def cf_palantir(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # ref: https://palantir.readthedocs.io/en/latest/notebooks/Palantir_sample_notebook.html
    # ref: https://scanpy.readthedocs.io/en/stable/external/generated/scanpy.external.tl.palantir.html
    # 1. prepare data
    adata = adata.copy()

    # 2. preprocess and execute method simutaneously with pca
    # don't filter cells
    # sc.pp.filter_cells(adata, min_counts=1000)
    # sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    sc.pp.pca(adata, n_comps=parameters["ndim"])
    sc.pp.neighbors(adata, knn=parameters["knn"])

    # 3. extract results
    sce.tl.palantir(adata, n_components=5, knn=parameters["knn"])
    pseudotime = np.zeros(adata.shape[0])

    # 4. save results
    trajectory_dict = {
        "pseudotime": adata.obs["palantir_pseudotime"],
    }

    return trajectory_dict

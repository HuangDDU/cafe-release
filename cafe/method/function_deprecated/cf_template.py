import anndata as ad
import numpy as np

# import scanpy as sc
# import scanpy.external as sce


def cf_template(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # 1. prepare data
    adata = adata.copy()

    # 2. preprocess and execute method simutaneously with pca
    pass

    # 3. extract results
    pseudotime = np.zeros(adata.shape[0])

    # 4. save results
    trajectory_dict = {
        "pseudotime": pseudotime,
    }

    return trajectory_dict

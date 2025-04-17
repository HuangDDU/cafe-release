import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import cellrank as cr


def cf_cellrank(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # 1. prepare data
    copy = parameters.get("copy", True)
    adata = adata.copy() if copy else adata
    cell_ids = adata.obs.index

    # 2. preprocess and execute method simutaneously with pca
    repreprocess = parameters.get("repreprocess", True)
    n_comps = parameters["ndim"]
    knn = knn = parameters["knn"]
    if repreprocess:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        sc.tl.pca(adata, random_state=0)
        sc.pp.neighbors(adata, random_state=0)

    # 3. extract results
    kernel = parameters.get("kernel", "connectivity")
    n_states = parameters.get("n_states", 10)
    n_terminal_states = parameters.get("n_terminal_states", 6)
    if kernel == "connectivity":
        # kernel
        ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
        ck.compute_transition_matrix()
        # estimator
        g = cr.estimators.GPCCA(ck)
        g.fit(n_states=n_states, cluster_key="clusters")
        g.predict_terminal_states(method="top_n", n_states=n_terminal_states)
        g.compute_fate_probabilities()
        # lineage object
        lineage = g._fate_probabilities
        end_state_probabilities = pd.DataFrame(lineage.__array__(), columns=lineage.names)
        end_state_probabilities["cell_id"] = cell_ids
    else:
        # TODO: Other kernel in parameters
        pass


    # 4. save results
    trajectory_dict = {
        "end_state_probabilities": end_state_probabilities,
    }

    return trajectory_dict

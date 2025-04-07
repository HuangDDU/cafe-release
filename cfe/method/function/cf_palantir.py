import numpy as np
import anndata as ad
import scanpy as sc
import scanpy.external as sce


def cf_palantir(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {},
    copy: bool = True,
    repreprocess: bool = True
):
    # ref: https://palantir.readthedocs.io/en/latest/notebooks/Palantir_sample_notebook.html
    # ref: https://scanpy.readthedocs.io/en/stable/external/generated/scanpy.external.tl.palantir.html
    # 1. prepare data
    adata = adata.copy() if copy else adata
    cell_ids = adata.obs.index

    # 2. preprocess and execute method simutaneously with pca
    n_comps = parameters["ndim"]
    knn = knn = parameters["knn"]
    if repreprocess:
        print("repreprocess")
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=n_comps)
        sc.pp.neighbors(adata, knn=knn)

    # 3. extract results
    sce.tl.palantir(
        adata,
        n_components=3,
        knn=knn
    )  # DiffusionMap and MAGIC
    early_cell = prior_information["start_id"]
    terminal_states = prior_information["terminal_states"]
    pr_res = sce.tl.palantir_results(
        adata,
        early_cell=early_cell,
        terminal_states=terminal_states
    )

    # 4. save results
    # multiple output data which adapt to multiple wrapper
    # TODO: multiple output wrapper parallelization
    # TODO: linear
    wrapper_type = parameters.get("wrapper_type", "linear")
    linear_type = parameters.get("linear_type", "pseudotime")
    if linear_type == "pseudotime":
        # pseudotime
        pseudotime = pr_res.pseudotime
    else:
        # entropy
        pseudotime = pr_res.entropy

    if wrapper_type == "linear":
        print("linear")
        trajectory_dict = {"pseudotime": pseudotime}
        return trajectory_dict
    else:
        # probability
        end_state_probabilities = pr_res.branch_probs
        end_state_probabilities["cell_id"] = cell_ids
        print(end_state_probabilities.shape)
        trajectory_dict = {
            "end_state_probabilities": end_state_probabilities,
            "pseudotime": pseudotime,
        }
        return trajectory_dict

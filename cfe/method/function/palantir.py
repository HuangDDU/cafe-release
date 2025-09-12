import anndata as ad

# import numpy as np
import scanpy as sc
import scanpy.external as sce


# TODO:
def palantir(
    adata: ad.AnnData,
    repreprocess: bool = True,
    palantir_kwargs: dict = {},
    palantir_results_kwargs: dict = {},
    n_comps: int = 5,
    knn: int = 30,
    wrapper_type: str = "linear",
    linear_type: str = "pseudotime",  # or entropy
    cluster_key: str = "clusters",
    **kwargs,
):
    # ref: https://palantir.readthedocs.io/en/latest/notebooks/Palantir_sample_notebook.html
    # ref: https://scanpy.readthedocs.io/en/stable/external/generated/scanpy.external.tl.palantir.html
    # 1. preprocess
    if repreprocess:
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=n_comps)
        sc.pp.neighbors(adata, knn=knn)
        print("repreprocess finish")

    # 2. execute method
    # TODO: check early_cell in cell_ids
    sce.tl.palantir(adata, **palantir_kwargs)  # DiffusionMap and MAGIC
    pr_res = sce.tl.palantir_results(adata, **palantir_results_kwargs)  # Pseudotime and branch probabilities
    print("palantir execute finish")

    # 3,4. extract and save results for different wrapper type
    # multiple output data which adapt to multiple wrapper
    # TODO: multiple output wrapper parallelization
    cell_ids = adata.obs.index
    if linear_type == "pseudotime":
        # pseudotime
        pseudotime = pr_res.pseudotime
    else:
        # entropy
        pseudotime = pr_res.entropy  # TODO:

    if wrapper_type == "linear":
        # for linear wrapper
        trajectory_dict = {"wrapper_type": "linear", "pseudotime": pseudotime}
    elif wrapper_type == "probability":
        # for probability wrapper
        end_state_probabilities = pr_res.branch_probs
        end_state_probabilities["cell_id"] = cell_ids
        trajectory_dict = {
            "end_state_probabilities": end_state_probabilities,
        }
    else:
        # TODO: for lineage wrapper
        terminal_states = palantir_results_kwargs.get("terminal_states", [])
        probability = pr_res.branch_probs
        probability.columns = adata.obs[cluster_key][cell_ids.get_indexer(terminal_states)]
        trajectory_dict = {
            "probability": probability,
            "cluster_key": cluster_key,
        }

    trajectory_dict["wrapper_type"] = wrapper_type

    return trajectory_dict

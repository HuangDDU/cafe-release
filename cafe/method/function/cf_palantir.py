from typing import Literal

import anndata as ad
import scanpy as sc
import scanpy.external as sce

try:
    from method_decorator import method_info
except ImportError:
    from cafe.method.function.method_decorator import method_info


@method_info(
    name="palantir",
    version="0.0.1",
    description="Palantir: characterization of cell fate probabilities",
    wrapper_type=["linear", "probability", "lineage"],
    doi="10.1038/s41587-019-0068-4",
    github_url="https://github.com/dpeerlab/Palantir",
    use_gpu=False,
    cpu_parallelization=True,
    available=True,
)
def palantir(
    adata: ad.AnnData,
    start_cell: str,
    repreprocess: bool = True,
    palantir_kwargs: dict = {},
    palantir_results_kwargs: dict = {},
    wrapper_type: Literal["linear", "probability", "lineage"] = "linear",
    linear_type: Literal["pseudotime", "entropy"] = "pseudotime",
    cluster: str = "clusters",
):
    """Palantir: characterization of cell fate probabilities

    Args:
        adata (ad.AnnData): The input AnnData object
        start_cell (str): The starting cell ID for palantir.
        repreprocess (bool, optional):  Whether to preprocess the data.
        palantir_kwargs (dict, optional): Palantir core parameter dict, refer to
            [scanpy.external.tl.palantir](https://scanpy.readthedocs.io/en/stable/external/generated/scanpy.external.tl.palantir.html).
        palantir_results_kwargs (dict, optional): Palantir result output parameter dict, refer to
            [scanpy.external.tl.palantir_results](https://scanpy.readthedocs.io/en/stable/external/generated/scanpy.external.tl.palantir_results.html).
        wrapper_type (Literal["linear", "probability", "lineage"], optional): Wrapper type for the output.
        linear_type (Literal["pseudotime", "entropy"], optional): Linear type for linear wrapper.
        cluster (str, optional): Cluster column in '.obs' columns for lineage wrapper.
    Returns:
        dict: A trajectory dict with keys: "wrapper_type" and "pseudotime".
    """

    # ref: https://palantir.readthedocs.io/en/latest/notebooks/Palantir_sample_notebook.html
    # ref: https://scanpy.readthedocs.io/en/stable/external/generated/scanpy.external.tl.palantir.html
    # 1. preprocess
    if repreprocess:
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        print("repreprocess finish")

    # 2. execute method
    # TODO: check early_cell in cell_ids
    sce.tl.palantir(adata, **palantir_kwargs)  # DiffusionMap and MAGIC
    pr_res = sce.tl.palantir_results(adata, early_cell=start_cell, **palantir_results_kwargs)  # Pseudotime and branch probabilities
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
        pseudotime = pr_res.entropy

    trajectory_dict = {}
    if wrapper_type == "linear":
        # for linear wrapper
        trajectory_dict["pseudotime"] = pseudotime
    elif wrapper_type == "probability":
        # for probability wrapper
        end_state_probabilities = pr_res.branch_probs
        end_state_probabilities["cell_id"] = cell_ids
        trajectory_dict["end_state_probabilities"] = end_state_probabilities
    else:
        # TODO: for lineage wrapper
        terminal_states = palantir_results_kwargs.get("terminal_states", [])
        probability = pr_res.branch_probs
        probability.columns = adata.obs[cluster][cell_ids.get_indexer(terminal_states)]

        trajectory_dict["probability"] = probability
        trajectory_dict["cluster"] = cluster

    trajectory_dict["wrapper_type"] = wrapper_type

    return trajectory_dict

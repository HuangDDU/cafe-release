import anndata as ad

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cfe environment
    from cfe.method.function.method_decorator import method_info
    from cfe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="phylovelo",
    version="0.0.1",
    description="PhyloVelo: Phylogeny-based transcriptomic velocity of single cells",
    wrapper_type="velocity",
    doi="10.1038/s41587-023-01887-5",
    github_url="https://github.com/kunwang34/PhyloVelo",
)
def phylovelo(
    adata: ad.AnnData,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = {},
):
    """PhyloVelo: Phylogeny-based transcriptomic velocity of single cells"""
    # 1. preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="scvelo", **repreprocess_kwargs)

    # 2. execute method

    # 3,4. extract and save results
    trajectory_dict = {
        "wrapper_type": "velocity",
        "velocity": adata.layers["velocity"],
        # "velocity_graph": adata.uns["velocity_graph"],
        # "velocity_graph_neg": adata.uns["velocity_graph_neg"],
        # "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        # "obs_index": adata.obs.index,
        # "var_index": adata.var.index,
    }

    return trajectory_dict

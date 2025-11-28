import anndata as ad

# import scvelo as scv

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cfe environment
    from cfe.method.function.method_decorator import method_info
    from cfe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="phlower",
    version="0.0.1",
    description="scVelo: PHLOWER leverages single-cell multimodal data to infer complex, multi-branching cell differentiation trajectories",
    wrapper_type="directed",
    doi="10.1038/s41592-025-02870-5",
    github_url="https://github.com/theislab/scvelo",
    use_gpu=False,
    cpu_parallelization=True,
)
def phlower(
    adata: ad.AnnData,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = {},
):
    """phlower: PHLOWER leverages single-cell multimodal data to infer complex, multi-branching cell differentiation trajectories

    Args:
        adata (ad.AnnData): AnnData object
        repreprocess (bool, optional): Whether to repreprocess the anndata object.

    Returns:
        dict: trajectory dict with keys about velocity
    """
    # ref: https://scvelo.readthedocs.io/en/stable/VelocityBasics.html
    # 1. preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="scvelo", **repreprocess_kwargs)

    # 2. execute method
    # scv.tl.velocity(adata, **velocity_kwargs)  # compute high dimensional velocity
    # scv.tl.velocity_graph(adata, **velocity_graph_kwargs)  # compute transition probability

    # scv.pl.velocity_embedding_stream(adata, basis="umap", show=False)  # don't plot here

    # 3,4. extract and save results
    trajectory_dict = {
        "wrapper_type": "velocity",
        # "velocity": adata.layers["velocity"],
        # "velocity_graph": adata.uns["velocity_graph"],
        # "velocity_graph_neg": adata.uns["velocity_graph_neg"],
        # "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        # "obs_index": adata.obs.index,
        # "var_index": adata.var.index,
    }
    return trajectory_dict

import anndata as ad
import scvelo as scv

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cafe environment
    from cafe.method.function.method_decorator import method_info
    from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="scvelo",
    version="0.0.1",
    description="scVelo: RNA velocity generalized through dynamical modeling",
    wrapper_type="velocity",
    doi="10.1038/s41587-020-0591-3",
    github_url="https://github.com/theislab/scvelo",
    use_gpu=False,
    cpu_parallelization=True,
)
def scvelo(
    adata: ad.AnnData,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = {},
    velocity_kwargs: dict = {},
    velocity_graph_kwargs: dict = {},
):
    """scVelo: RNA velocity generalized through dynamical modeling

    Args:
        adata (ad.AnnData): AnnData object
        repreprocess (bool, optional): Whether to repreprocess the anndata object.
        repreprocess_kwargs (dict, optional): Parameter dict for repreprocess pipeline.
        velocity_kwargs (dict, optional): Parameter dict for velocity calculation, refer to [scvelo.tl.velocity](https://scvelo.readthedocs.io/en/stable/scvelo.tl.velocity.html).
        velocity_graph_kwargs (dict, optional): Parameter dict for velocity graph calculation, refer to [scvelo.tl.velocity_embedding](https://scvelo.readthedocs.io/en/stable/scvelo.tl.velocity_embedding.html).

    Returns:
        dict: trajectory dict with keys about velocity
    """
    # ref: https://scvelo.readthedocs.io/en/stable/VelocityBasics.html
    # 1. preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="scvelo", **repreprocess_kwargs)

    # 2. execute method
    scv.tl.velocity(adata, **velocity_kwargs)  # compute high dimensional velocity
    scv.tl.velocity_graph(adata, **velocity_graph_kwargs)  # compute transition probability
    # scv.pl.velocity_embedding_stream(adata, basis="umap", show=False)  # don't plot here
    adata.uns["method_name"] = scvelo  # to find correspodding function "extract_trajectory_dict" easily

    # 3,4. extract and save results
    trajectory_dict = extract_trajectory_dict(adata)
    return trajectory_dict


def extract_trajectory_dict(adata):
    trajectory_dict = {
        "wrapper_type": "velocity",
        "velocity": adata.layers["velocity"],
        "velocity_graph": adata.uns["velocity_graph"],
        "velocity_graph_neg": adata.uns["velocity_graph_neg"],
        "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,
    }
    return trajectory_dict

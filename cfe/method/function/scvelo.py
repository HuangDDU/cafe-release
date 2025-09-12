import anndata as ad
import scvelo as scv


def scvelo(
    adata: ad.AnnData,
    repreprocess: bool = True,
    filter_and_normalize_kwargs: dict = {},
    moments_kwargs: dict = {},
    velocity_kwargs: dict = {},
    velocity_graph_kwargs: dict = {},
):
    """scVelo for RNA Velcoity

    Args:
        adata (ad.AnnData): AnnData object
        repreprocess (bool, optional): whether reprocess the anndata object, including feature selection, normalization, scale, pca and neighbor computation. Defaults to True.
        filter_and_normalize_kwargs (dict, optional): Parameters for preprocess in scvelo style, refer to "scvelo.pp.filter_and_normalize"(https://scvelo.readthedocs.io/en/stable/scvelo.pp.filter_and_normalize.html). Defaults to {}.
        moments_kwargs (dict, optional): Parameters for neighbor moment in scvelo style, refer to "scvelo.pp.moments"(https://scvelo.readthedocs.io/en/stable/scvelo.pp.moments.html). Defaults to {}.
        velocity_kwargs (dict, optional): refer to https://scvelo.readthedocs.io/en/stable/scvelo.tl.velocity.html. Defaults to {}.
        velocity_graph_kwargs (dict, optional): refer to https://scvelo.readthedocs.io/en/stable/scvelo.tl.velocity_embedding.html. Defaults to {}.

    Returns:
        dict: trajectory dict
    """
    # ref: https://scvelo.readthedocs.io/en/stable/VelocityBasics.html
    # 1. preprocess
    if repreprocess:
        scv.pp.filter_and_normalize(adata, **filter_and_normalize_kwargs)
        scv.pp.moments(adata, **moments_kwargs)

    # 2. execute method
    scv.tl.velocity(adata, **velocity_kwargs)  # compute high dimensional velocity
    scv.tl.velocity_graph(adata, **velocity_graph_kwargs)  # compute transition probability

    # scv.pl.velocity_embedding_stream(adata, basis="umap", show=False)  # don't plot here

    # 3,4. extract and save results
    trajectory_dict = {
        "velocity": adata.layers["velocity"],
        "velocity_graph": adata.uns["velocity_graph"],
        "velocity_graph_neg": adata.uns["velocity_graph_neg"],
        "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,
    }
    return trajectory_dict

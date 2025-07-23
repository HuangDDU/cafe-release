import anndata as ad
import scvelo as scv


def cf_scvelo(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # ref: https://scvelo.readthedocs.io/en/stable/VelocityBasics.html
    # 1. extract prior information and parameters
    repreprocess = parameters["repreprocess"]
    filter_and_normalize_kwargs = parameters["filter_and_normalize_kwargs"]
    moments_kwargs = parameters["moments_kwargs"]
    velocity_kwargs = parameters["velocity_kwargs"]
    velocity_graph_kwargs = parameters["velocity_graph_kwargs"]

    # 2. preprocess
    if repreprocess:
        scv.pp.filter_and_normalize(adata, **filter_and_normalize_kwargs)
        scv.pp.moments(adata, **moments_kwargs)

    # 3. execute method
    scv.tl.velocity(adata, **velocity_kwargs)  # compute high dimensional velocity
    scv.tl.velocity_graph(adata, **velocity_graph_kwargs)  # compute transition probability

    # scv.pl.velocity_embedding_stream(adata, basis="umap", show=False)  # don't plot here

    # 4,5. extract and save results
    trajectory_dict = {
        "velocity": adata.layers["velocity"],
        "velocity_graph": adata.uns["velocity_graph"],
        "velocity_graph_neg": adata.uns["velocity_graph_neg"],
        "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,
    }
    return trajectory_dict

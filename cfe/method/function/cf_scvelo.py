import anndata as ad
import scvelo as scv


def cf_scvelo(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # ref: https://scvelo.readthedocs.io/en/stable/VelocityBasics.html
    # cluster_key = prior_information.get("cluster_key", "clusters")

    # 1. prepare data
    adata = adata.copy()

    # 2. preprocess
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # 3. execute method
    scv.tl.velocity(adata)  # compute high dimensional velocity
    scv.tl.velocity_graph(adata)  # compute transition probability
    # scv.pl.velocity_embedding_stream(adata, basis="umap", show=False)  # don't plot here

    # 4. extract results
    trajectory_dict = {
        "velocity": adata.layers["velocity"],
        "velocity_graph": adata.uns["velocity_graph"],
        "velocity_graph_neg": adata.uns["velocity_graph_neg"],
        "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,
    }
    return trajectory_dict

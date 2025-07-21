import anndata as ad
import scvelo as scv


def cf_velovi(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # the package is not available in cfe envirionment, so we import it here
    import scvi

    # TODO: optimize parameter for better result.
    # cluster_key = prior_information.get("cluster_key", "clusters")  # do nothing, only prior information demo
    max_epochs = parameters.get("max_epochs", 50)

    # 1. prepare data
    adata = adata.copy()

    # 2. preprocess
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # 3. execute method
    VELOVI = scvi.external.VELOVI  # extract the VELOVI class
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train(max_epochs=max_epochs)  # TODO: very slow, need GPU
    # extract velocity to adata.layers["velocity"]
    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")
    t = latent_time
    scaling = 20 / t.max(0)
    adata.layers["velocity"] = velocities / scaling
    scv.tl.velocity_graph(adata)

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

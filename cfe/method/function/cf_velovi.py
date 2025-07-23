import anndata as ad
import scvelo as scv


def cf_velovi(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # ref: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/velovi.html
    # the package is not available in cfe envirionment, so we import it here
    import scvi

    # 1. extract prior information and parameters
    # cluster_key = prior_information.get("cluster_key", "clusters")  # do nothing, only prior information demo
    repreprocess = parameters["repreprocess"]
    filter_and_normalize_kwargs = parameters["filter_and_normalize_kwargs"]
    moments_kwargs = parameters["moments_kwargs"]
    velovi_model_kwargs = parameters["velovi_model_kwargs"]
    velovi_train_kwargs = parameters["velovi_train_kwargs"]
    n_sample = parameters["n_sample"]

    # 2. preprocess
    if repreprocess:
        scv.pp.filter_and_normalize(adata, **filter_and_normalize_kwargs)
        scv.pp.moments(adata, **moments_kwargs)

    # 3. execute method
    VELOVI = scvi.external.VELOVI  # extract the VELOVI class
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata, **velovi_model_kwargs)
    vae.train(**velovi_train_kwargs)  # TODO: very slow, need GPU
    # extract velocity to adata.layers["velocity"]
    latent_time = vae.get_latent_time(n_samples=n_sample)
    velocities = vae.get_velocity(n_samples=n_sample, velo_statistic="mean")
    t = latent_time
    scaling = 20 / t.max(0)
    adata.layers["velocity"] = velocities / scaling
    scv.tl.velocity_graph(adata)

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

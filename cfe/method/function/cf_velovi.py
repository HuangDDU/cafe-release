import anndata as ad
import scvelo as scv

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cfe environment
    from cfe.method.function.method_decorator import method_info
    from cfe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="velovi",
    version="0.0.1",
    description="VeloVI: Deep generative modeling of transcriptional dynamics for RNA velocity analysis in single cells",
    wrapper_type="velocity",
    doi="10.1038/s41592-023-01994-w",
    github_url="https://github.com/yoseflab/velovi",
)
def velovi(
    adata: ad.AnnData,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = {},
    velovi_model_kwargs: dict = {},
    velovi_train_kwargs: dict = {},
    n_sample: int = 25,
):
    """Deep generative modeling of transcriptional dynamics for RNA velocity analysis in single cells
    Args:
        adata (ad.AnnData): AnnData object
        repreprocess (bool, optional):  Whether to repreprocess the anndata object.
        repreprocess_kwargs (dict, optional):  Parameter dict for repreprocess pipeline.
        velovi_model_kwargs (dict, optional): Parameter dict for VeloVI model initialization, refer to [scvi.external.VELOVI](https://docs.scvi-tools.org/en/stable/api/reference/scvi.external.VELOVI.html).
        velovi_train_kwargs (dict, optional): Parameter dict for VeloVI model training , refer to [scvi.external.VELOVI.train](https://docs.scvi-tools.org/en/stable/api/reference/scvi.external.VELOVI.html#scvi.external.VELOVI.train).
        n_sample (int, optional): Sample number from latent space, refer to [scvi.external.VELOVI.get_latent_time](https://docs.scvi-tools.org/en/stable/api/reference/scvi.external.VELOVI.html#scvi.external.VELOVI.get_latent_time).

    Returns:
        dict: trajectory dict with keys about velocity

    """
    # ref: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/velovi.html
    # the package is not available in cfe envirionment, so we import it here
    from scvi.external import VELOVI

    # 1. preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="scvelo", **repreprocess_kwargs)

    # 2. execute method
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

    # 3,4. extract and save results
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

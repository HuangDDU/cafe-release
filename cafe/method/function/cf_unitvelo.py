import anndata as ad

try:
    # for docker
    from method_decorator import method_info

    # from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cafe environment
    from cafe.method.function.method_decorator import method_info

    # from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="unitvelo",
    version="0.0.1",
    description="UniTVelo: temporally unified RNA velocity reinforces single-cell trajectory inference",
    wrapper_type="velocity",
    doi="10.1038/s41467-022-34188-7",
    github_url="https://github.com/StatBiomed/UniTVelo",
)
def unitvelo(
    adata: ad.AnnData,
    cluster: str,
    configuration_kwargs: dict = {},
):
    """UniTVelo: temporally unified RNA velocity reinforces single-cell trajectory inference

    Args:
        adata (ad.AnnData): AnnData object.
        cluster (str): Cluster column name in adata.obs.
        configuration_kwargs (dict, optional):  Parameter dict for unitvelo pipeline, refer to [config.py](https://github.com/StatBiomed/UniTVelo/blob/main/unitvelo/config.py).

    Returns:
        dict: trajectory dict with keys about velocity
    """

    # ref: https://github.com/StatBiomed/UniTVelo/blob/main/notebooks/Figure3_BoneMarrow.ipynb
    import os
    import shutil

    import unitvelo as utv

    # 1,2 preprocess and execute method
    if "filename" in adata.uns:
        adata_filename = adata.uns["filename"]
    else:
        # for docker test: save adata for latter unitvelo pipeline
        adata_filename = "adata.h5ad"
        adata.write(adata_filename)
        print("save adata for unitvelo pipeline:", adata_filename)

    # configuration
    velo = utv.config.Configuration()
    velo.MAX_ITER = 1000
    for k, v in configuration_kwargs.items():
        setattr(velo, k, v)
    # run model
    adata = utv.run_model(adata_filename, label=cluster, config_file=velo)

    # remove tmp dir
    tmp_dir = adata_filename.replace(".h5ad", "")
    if os.path.exists(tmp_dir):
        print("remove unitvelo tmp dir:", tmp_dir)
        shutil.rmtree(tmp_dir)

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

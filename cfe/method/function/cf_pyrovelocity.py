import anndata as ad

try:
    # for docker
    from method_decorator import method_info

    # from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cfe environment
    from cfe.method.function.method_decorator import method_info

    # from cfe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="pyrovelocity",
    version="0.0.1",
    description="PyroVelocity: probabilistic modeling of RNA velocity",
    wrapper_type="velocity",
    doi="10.1101/2022.09.12.507691",
    github_url="https://github.com/pinellolab/pyrovelocity",
)
def pyrovelocity(
    adata: ad.AnnData,
    repreprocess: bool = True,  # unused
    # repreprocess_kwargs: dict = {},
):
    """PyroVelocity: probabilistic modeling of RNA velocity"""
    #  ref: https://docs.pyrovelocity.net/templates/user_example/user_example
    #  pyrovelocity workflow add.
    import tempfile

    import mlflow
    import scanpy as sc
    from pyrovelocity.workflows.main_configuration import (
        pancreas_configuration as templete_configuration,
    )
    from pyrovelocity.workflows.main_workflow import (
        download_data,
        postprocess_data,
        preprocess_data,
        train_model,
    )

    with tempfile.TemporaryDirectory() as tmp_wd:
        #
        data_set_name = "tmp"
        adata_filename = f"{tmp_wd}/{data_set_name}.h5ad"
        sc.write(adata_filename, adata)

        # configuration object
        templete_configuration.download_dataset.data_set_name = data_set_name
        templete_configuration.download_dataset.data_external_path = tmp_wd
        templete_configuration.download_dataset.source = ""

        templete_configuration.preprocess_data.data_set_name = data_set_name
        templete_configuration.preprocess_data.adata = adata_filename

        templete_configuration.training_configuration_1.data_set_name = data_set_name
        templete_configuration.training_configuration_1.adata = f"{tmp_wd}/{data_set_name}_processed.h5ad"
        templete_configuration.training_configuration_1.max_epochs = 200

        # data
        data = download_data(download_dataset_args=templete_configuration.download_dataset)

        # preprocess
        processed_data = preprocess_data(
            data=data,
            preprocess_data_args=templete_configuration.preprocess_data,
        )

        # train
        mlflow.set_experiment("0")
        model_output = train_model(
            processed_data,
            train_model_configuration=templete_configuration.training_configuration_1,
        )

        # postprocess
        postprocessing_outputs = postprocess_data(
            preprocess_data_args=templete_configuration.preprocess_data,
            training_outputs=model_output,
            postprocess_configuration=templete_configuration.postprocess_configuration,
        )

        # read result adata
        adata = sc.read(postprocessing_outputs.postprocessed_data)

    trajectory_dict = {
        "wrapper_type": "velocity",
        "velocity": adata.layers["velocity_pyro"],
        "velocity_graph": adata.uns["velocity_graph"],
        "velocity_graph_neg": adata.uns["velocity_graph_neg"],
        "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,
    }

    return trajectory_dict

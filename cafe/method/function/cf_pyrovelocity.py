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
    name="pyrovelocity",
    version="0.0.1",
    description="PyroVelocity: probabilistic modeling of RNA velocity",
    wrapper_type="velocity",
    doi="10.1101/2022.09.12.507691",
    github_url="https://github.com/pinellolab/pyrovelocity",
    use_gpu=True,
    cpu_parallelization=True,
    available=True,
)
def pyrovelocity(
    adata: ad.AnnData,
    configuration_kwargs: dict = {},
):
    """PyroVelocity: probabilistic modeling of RNA velocity

    Args:
        adata (ad.AnnData): AnnData object.
        configuration_kwargs (dict, optional): Configuration dict for pyrovelocity pipeline, refer to [pancrease template](https://github.com/pinellolab/pyrovelocity/blob/v0.4.5/src/pyrovelocity/workflows/main_configuration.py).

    Returns:
        dict: trajectory dict with keys about velocity
    """

    #  ref: https://docs.pyrovelocity.net/templates/user_example/user_example
    #  pyrovelocity workflow add.
    # https://github.com/pyrovelocity/pyrovelocity/blob/v0.4.5/src/pyrovelocity/workflows/main_configuration.py
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

    # working dir is setting to avoid use previous middle file.

    with tempfile.TemporaryDirectory() as tmp_wd:
        #
        data_set_name = "tmp"
        # adata_filename = f"{tmp_wd}/{data_set_name}.h5ad"
        # sc.write(adata_filename, adata)
        if "filename" in adata.uns:
            adata_filename = adata.uns["filename"]
        else:
            adata_filename = "adata.h5ad"
            adata.write(adata_filename)
            print("save adata for pyrovelocity pipeline:", adata_filename)
        data_set_name = adata_filename.split("/")[-1]
        data_external_path = adata_filename.replace(data_set_name, "")
        data_set_name = data_set_name.replace(".h5ad", "")

        # configuration object construction based on pancrease template
        # input, preprocessed and output filename
        templete_configuration.download_dataset.data_set_name = data_set_name
        templete_configuration.download_dataset.data_external_path = data_external_path
        templete_configuration.download_dataset.source = ""
        templete_configuration.preprocess_data.data_set_name = data_set_name
        templete_configuration.preprocess_data.adata = adata_filename
        templete_configuration.training_configuration_1.data_set_name = data_set_name
        templete_configuration.training_configuration_1.adata = f"{tmp_wd}/{data_set_name}_processed.h5ad"
        # other configuration
        templete_configuration.training_configuration_1.max_epochs = 200
        for category, category_configuration_dict in configuration_kwargs.items():
            category_configuration_object = getattr(templete_configuration, category)
            if category_configuration_object is None:
                print(f"Warning: no category '{category}' in template configuration")
                continue
            else:
                for k, v in category_configuration_dict.items():
                    if hasattr(category_configuration_object, k):
                        setattr(category_configuration_object, k, v)
                    else:
                        print(f"Warning: no parameter '{k}' in configuration category '{category}'")

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
        print("adata result", adata.shape)

    trajectory_dict = {
        "wrapper_type": "velocity",
        "velocity": adata.layers["velocity_pyro"],
        "velocity_graph": adata.uns["velocity_pyro_graph"],
        "velocity_graph_neg": adata.uns["velocity_pyro_graph_neg"],
        "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,
        "save_h5ad": postprocessing_outputs.postprocessed_data,
    }

    return trajectory_dict

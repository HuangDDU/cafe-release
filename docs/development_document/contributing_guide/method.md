# Method

> This article will use the addition of a classic velocity wrapper `velovi` as an example to illustrate how developers can contribute new trajectory inference methods to this project. You should develop new trajectory inference method in `cfe/method` directory. Additionally, test case, document and environment management are required.

> You can see the file change example in git commit `73f6ba1978a0677292cdc84b1f3933ce6ff9586f`.

## Register fate method and create environment

1. Ensure you have local conda environment `scvi-env`, istructed by [scvi-tools installation document](https://docs.scvi-tools.org/en/stable/installation.html).
2. Register the method in backend config file: `cfe/method/method_backend.yml`, including method name, python function and conda environment.

    ``` yaml
    velovi:
        python_function: cf_velovi
        conda_env: scvi-env
    ```

3. Craete the method config file: `cfe/method/definition/velovi.yml`, including method metadata, wapper type ,parameters and so on.

    ```yaml
    method:
    id: velovi
    name: veloVI
    version: 0.0.1
    tool_id: velovi
    source: tool
    platform: Python
    url: https://github.com/yoseflab/velovi

    wrapper:
    type: velocity
    input_required:
    - counts
    input_optional:
    - cluster_key # prior_information

    # TODO: add docker container
    container:
    docker: cfe/cf_velovi

    manuscript:
    doi: doi.org/10.1038/s41592-023-01994-w

    parameters:
    - id: max_epochs
    description: Max number of epochs for velocity parameters estimation.
    type: integer
    default: 10
    ```

## Python function

Create the method file `cfe/method/function/cf_scvelo.py` as following:

```python
import anndata as ad
import scvelo as scv


def cf_velovi(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    import scvi

    # import torch
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
```

## Test in jupyter notebook

1. Unit test in pytest script:
    - Create pytest script `tests/method/function/test_cf_velovi.py`the pytest marker `run_method`, this test using common test data.

        ```python
        import os
        import sys

        import pytest
        import scanpy as sc

        # don't import cfe here, because other conda environment may not have cfe relative package file

        # Note: run the following command to run the test  in project dir with scvi-env conda environment
        # conda activate scvi-env
        # pytest -s --tb=long  --collect-only --run-method tests/method/function/test_cf_velovi.py

        sys.path.append("../../../cfe/method/function")  # prepare relative package file for  file dir
        sys.path.append("cfe/method/function")  # prepare relative package file for project dir


        class TestCFVeloVI:
            def setup_method(self):
                self.adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")

            @pytest.mark.run_method
            def test_velovi(self):
                from cf_velovi import cf_velovi

                # add priority and parameters
                prior_information = {"cluster_key": "lineage"}
                parameters = {"max_epochs": 1}
                trajectory_dict = cf_velovi(self.adata, prior_information, parameters)
                assert trajectory_dict.keys() == {"velocity", "velocity_graph", "velocity_graph_neg", "neighbors", "obs_index", "var_index"}


        if __name__ == "__main__":
            pytest.main(["-v", __file__])
        ```

    - You can add `-m run_method` in pytest command to run this test markered as run_method. However, vscode automatical test will skip it.

        ``` bash
        # pwd: project directory
        pytest -s --tb=long  --collect-only --run-method tests/method/function/test_cf_velovi.py
        ```

    - The unit test only check the method's successful run and output trajectory dict format. However, the accuracy of the output trajectory dict is not guaranteed, you need to check the accuracy in intergrated test by notebook

2. Intergrated test in jupyter notbeook:
    - create and run notebook `notebook_dev/hzy/dev_scvelo_pancreas_500_conda.ipynb`.
    - check the visualization result.

## Submit conda environment(TODO)

1. Copy the conda environment to common conda directory

    ```bash
    # sudo
    cp ~/.conda/envs/scvi-env /usr/local/conda/envs/scvi-env 

    ```

2. Export conda yaml file and share it.

## Docker environment(TODO)

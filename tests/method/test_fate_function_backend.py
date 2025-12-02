import os.path

import pytest
import scanpy as sc

import cafe


def get_test_run_data():
    # ref: notebook/quickstart_paga.ipynb, reuse for other backend
    adata = sc.read(f"{os.path.dirname(__file__)}/../data/bifurcating.h5ad")
    fadata = cafe.data.FateAnnData.from_anndata(adata)
    fadata.layers["counts"] = fadata.X.copy()
    fadata.layers["expression"] = fadata.X.copy()
    fadata.obs.index = fadata.obs["cell_id"]
    # prior_information,  parameters
    prior_information = {"start_id": "cell1"}
    parameters = {"cluster_key": "lineage"}
    fadata.add_prior_information(**prior_information)  # add prior information to fadata

    return fadata, parameters


function_name = "comp1"


class TestFunctionBackend:
    def setup_method(self):
        self.function_backend = cafe.method.FunctionBackend(function_name)

    def test_load_backend(self):
        assert self.function_backend.function_name == function_name

    def test_run(self):
        fadata, parameters = get_test_run_data()
        self.function_backend.run(fadata, parameters)
        assert fadata.is_wrapped_with_trajectory

    # # TOOD: consider if __call__ is needed
    # def test_call(self):
    #     adata = sc.read(f"{os.path.dirname(__file__)}/../data/bifurcating.h5ad")
    #     adata.obs.index = adata.obs["cell_id"]

    #     parameters = {
    #         "start_id": "cell1",
    #         "cluster_key": "lineage",
    #         "connectivity_cutoff": 0.5,
    #     }

    #     self.function_backend(adata, **parameters)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

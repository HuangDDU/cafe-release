import os.path

import pytest
import scanpy as sc

import cfe

method_name = "comp1"
# method_name = "paga"


class TestFateMethod:
    def setup_method(self):
        cfe.settings.backend = "python_function"
        self.fate_method = cfe.method.FateMethod(method_name=method_name)

    def test_init(self):
        fate_method = self.fate_method
        assert fate_method.method_name == method_name
        assert fate_method.backend_name is None
        assert fate_method.backend is None

    def test_choose_backend(self):
        fate_method = self.fate_method

        fate_method.choose_backend(backend="python_function")
        assert fate_method.backend == "python_function" and isinstance(fate_method.method_backend, cfe.method.FunctionBackend)

        self.fate_method.choose_backend(backend="cfe_docker")
        assert fate_method.backend == "cfe_docker" and isinstance(fate_method.method_backend, cfe.method.CFEDockerBackend)

        if cfe.settings.r_available:
            # test dynverse docker when R is available
            self.fate_method.choose_backend(backend="dynverse_docker")
            assert fate_method.backend == "dynverse_docker" and isinstance(fate_method.method_backend, cfe.method.DynverseDockerBackend)

    def test_infer_trajectory(self):
        # notebook/quickstart_paga.ipynb
        # data
        # TODO：Method, Backend测试样例通过
        adata = sc.read(f"{os.path.dirname(__file__)}/../data/bifurcating.h5ad")
        fadata = cfe.data.FateAnnData.from_anndata(adata)
        fadata.layers["counts"] = fadata.X.copy()
        fadata.layers["expression"] = fadata.X.copy()
        fadata.obs.index = fadata.obs["cell_id"]
        # add prior information to fadata
        prior_information = {"clusters": "lineage", "start_id": "cell1"}
        fadata.add_prior_information(**prior_information)
        parameters = {}

        self.fate_method.infer_trajectory(fadata, parameters)

        assert fadata.is_wrapped_with_trajectory

    # TOOD: consider if __call__ is needed
    # def test_call(self):
    #     adata = sc.read(f"{os.path.dirname(__file__)}/../data/bifurcating.h5ad")
    #     fadata = cfe.data.FateAnnData.from_anndata(adata)
    #     fadata.obs.index = fadata.obs["cell_id"]

    #     parameters = {
    #         "start_id": "cell1",
    #         "cluster_key": "lineage",
    #         "connectivity_cutoff": 0.5,
    #     }
    #     self.fate_method(fadata, **parameters)  # call __call__

    #     assert fadata.is_wrapped_with_trajectory

    # TOOD: consider if __call__ is needed
    # def test_get_parameter_df(self):
    #     self.fate_method.choose_backend(backend="python_function")
    #     parameter_df = self.fate_method.get_parameter_df()

    #     assert isinstance(parameter_df, pd.DataFrame)
    #     assert list(parameter_df.columns) == ["description", "type", "default", "update", "distribution"]

    # def test_get_prior_information_df(self):
    #     prior_information_df = self.fate_method.get_prior_information_df()


if __name__ == "__main__":
    pytest.main(["-v", __file__])

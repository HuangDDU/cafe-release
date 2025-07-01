import os.path

import pytest
import scanpy as sc

import cfe


class TestFateMethod:
    def setup_method(self):
        cfe.settings.backend = "python_function"
        self.fate_method = cfe.method.FateMethod(method_name="paga")

    def test_init(self):
        fate_method = self.fate_method
        assert fate_method.method_name == "paga"
        assert fate_method.backend == "python_function"

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
        adata = sc.read(f"{os.path.dirname(__file__)}/../data/bifurcating.h5ad")
        fadata = cfe.data.FateAnnData.from_anndata(adata)
        fadata.layers["counts"] = fadata.X.copy()
        fadata.layers["expression"] = fadata.X.copy()
        cluster_key = "lineage"
        fadata.obs.index = fadata.obs["cell_id"]
        # prior_information,  parameters
        prior_information = {"start_id": "cell1", "groups_id": fadata.obs[cluster_key].tolist()}
        parameters = {"filter_features": False}
        fadata.add_prior_information(**prior_information)  # add prior information to fadata

        self.fate_method.infer_trajectory(fadata, parameters)

        assert fadata.is_wrapped_with_trajectory


if __name__ == "__main__":
    pytest.main(["-v", __file__])

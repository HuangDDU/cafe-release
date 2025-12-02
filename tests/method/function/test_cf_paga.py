import os

import pytest
import scanpy as sc

import cafe

from .method_testcase import method_testcase


class TestCFPAGA:
    def setup_method(self):
        self.method_name = "paga"

        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/pancrease_scvelo_500_fadata.h5ad")
        fadata = cafe.data.FateAnnData.from_anndata(adata)

        self.adata = adata
        self.fadata = fadata
        self.parameters = {
            "repreprocess": True,
            "start_cell": "cell_000",
            "cluster": "clusters",
            "n_dcs": 2,
            "connectivity_cutoff": 0.5,
        }

    # Test raw trajectory dict
    def test_raw(self):
        # call function directly, use AnnData
        from cafe.method.function.cf_paga import paga

        trajectory_dict = paga(self.adata, **self.parameters)
        assert trajectory_dict.keys() == {"wrapper_type", "branch_network", "branches", "branch_progressions"}  # check trajectory dict keys

    # Test three backends
    def test_function(self):
        fadata = method_testcase(self.adata, self.method_name, "python_function", self.parameters)
        assert fadata.is_wrapped_with_trajectory

    def test_conda(self):
        fadata = method_testcase(self.adata, self.method_name, "conda", self.parameters)
        assert fadata.is_wrapped_with_trajectory

    def test_docker(self):
        fadata = method_testcase(self.adata, self.method_name, "cafe_docker", self.parameters)
        assert fadata.is_wrapped_with_trajectory


if __name__ == "__main__":
    pytest.main(["-v", __file__])

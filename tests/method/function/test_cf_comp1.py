import os

import pytest
import scanpy as sc

import cfe

from .method_testcase import method_testcase


class TestCFComp1:
    def setup_method(self):
        self.method_name = "comp1"

        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/pancrease_scvelo_500_fadata.h5ad")
        fadata = cfe.data.FateAnnData.from_anndata(adata)

        self.adata = adata
        self.fadata = fadata
        self.parameters = {"repreprocess": True, "basis": "X_pca", "component": 1}

    # Test raw trajectory dict
    def test_raw(self):
        # call function directly, use AnnData
        from cfe.method.function.cf_comp1 import comp1

        trajectory_dict = comp1(self.adata, **self.parameters)
        assert trajectory_dict.keys() == {"pseudotime", "wrapper_type"}  # check trajectory dict keys

    # Test three backends
    def test_function(self):
        fadata = method_testcase(self.adata, self.method_name, "python_function", self.parameters)
        assert fadata.is_wrapped_with_trajectory

    def test_conda(self):
        fadata = method_testcase(self.adata, self.method_name, "conda", self.parameters)
        assert fadata.is_wrapped_with_trajectory

    def test_docker(self):
        fadata = method_testcase(self.adata, self.method_name, "cfe_docker", self.parameters)
        assert fadata.is_wrapped_with_trajectory

    def test_call(self):
        # FateMethod.__call__
        cfe.method.cf_comp1(self.fadata, self.parameters)
        assert self.fadata.is_wrapped_with_trajectory


if __name__ == "__main__":
    pytest.main(["-v", __file__])

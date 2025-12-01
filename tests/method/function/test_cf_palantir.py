import os

import pytest
import scanpy as sc

from .method_testcase import method_testcase

if_test_raw = False  # change to True when run in 'palantir' conda environment


class TestCFPalantir:
    def setup_method(self):
        self.method_name = "palantir"
        self.adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/pancrease_scvelo_500_fadata.h5ad")  # need real data for gene name
        self.parameters = {"repreprocess": True, "start_cell": "cell_000"}

    # Test raw trajectory dict
    # conda activate sctc
    # pytest -s --tb=long test_cf_palantir.py
    @pytest.mark.skipif(not if_test_raw, reason="skip raw test, because it should be in conda environment 'sctc'")
    def test_raw(self):
        import sys

        sys.path.append("../../../cafe/method/function")  # prepare relative package file
        from cf_palantir import sctc

        trajectory_dict = sctc(self.adata, **self.parameters)

        assert trajectory_dict.keys() == {"pseudotime", "wrapper_type"}  # check trajectory dict keys

    # Test three backends
    # function backend is not available
    # def test_function(self):
    #     pass

    @pytest.mark.skipif(if_test_raw, reason="skip conda backend test, because it should be in conda environment 'cafe'")
    def test_conda(self):
        fadata = method_testcase(self.adata, self.method_name, "conda", self.parameters)
        assert fadata.is_wrapped_with_trajectory

    @pytest.mark.skipif(if_test_raw, reason="skip conda backend test, because it should be in conda environment 'cafe'")
    def test_docker(self):
        fadata = method_testcase(self.adata, self.method_name, "cafe_docker", self.parameters)
        assert fadata.is_wrapped_with_trajectory


if __name__ == "__main__":
    pytest.main(["-v", __file__])

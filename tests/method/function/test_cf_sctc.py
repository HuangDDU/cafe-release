import os
import sys

import pytest
import scanpy as sc

# pytest -s --tb=long -m run_method test_cf_sctc.py
sys.path.append("../../../cfe/method/function")  # prepare relative package file for  file dir


@pytest.mark.run_method
class TestCFSCTC:
    def setup_method(self):
        self.adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/pancrease_scvelo_500_fadata.h5ad")

    def test_sctc(self):
        from cf_sctc import cf_sctc

        trajectory_dict = cf_sctc(self.adata)
        assert trajectory_dict.keys() == {"pseudotime"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

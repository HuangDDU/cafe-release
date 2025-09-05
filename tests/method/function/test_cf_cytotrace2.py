import os
import sys

import pytest
import scanpy as sc

# pytest -s --tb=long -m run_method test_cf_cytotrace2.py
sys.path.append("../../../cfe/method/function")  # prepare relative package file for  file dir


@pytest.mark.run_method
class TestCFCytotrace2:
    def setup_method(self):
        self.adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/pancrease_scvelo_500_fadata.h5ad")

    def test_comp1_old(self):
        from cf_cytotrace2 import cf_cytotrace2

        prior_information = {}
        parameters = {}
        trajectory_dict = cf_cytotrace2(self.fadata, prior_information, parameters)
        assert trajectory_dict.keys() == {"pseudotime"}

    def test_cytotrace_new(self):
        from cf_cytotrace2 import cf_cytotrace2

        trajectory_dict = cf_cytotrace2(self.adata)
        assert trajectory_dict.keys() == {"pseudotime"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

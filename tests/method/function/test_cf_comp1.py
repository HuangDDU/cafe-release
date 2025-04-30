import pytest
import cfe

import os
import scanpy as sc


class TestCFComp1():
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_comp1(self):
        # add priority and parameeters
        prior_information = {}
        parameters = {"ndim": 2, "component": 1}
        trajectory_dict = cfe.method.cf_comp1(self.fadata, prior_information, parameters)
        assert trajectory_dict.keys() == {"pseudotime"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

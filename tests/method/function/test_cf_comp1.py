import os

import pytest
import scanpy as sc

import cfe


class TestCFComp1:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_comp1_old(self):
        # old version call: you need to provide priority and parameters besides adata
        # add priority and parameters
        prior_information = {}
        parameters = {"repreprocess": True, "pca_ndim": 10, "basis": "X_pca", "component": 1}
        trajectory_dict = cfe.method.cf_comp1(self.fadata, prior_information, parameters)
        assert "pseudotime" in trajectory_dict.keys()

    def test_comp1_new(self):
        # new version: you need to provide nothing beside adata, where default parameter are available.
        trajectory_dict = cfe.method.cf_comp1(self.fadata)
        assert "pseudotime" in trajectory_dict.keys()


if __name__ == "__main__":
    pytest.main(["-v", __file__])

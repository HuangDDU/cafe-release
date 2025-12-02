import os

import pytest
import scanpy as sc

import cafe


class TestCFAngle:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cafe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_angle_old(self):
        # add priority and parameters
        prior_information = {}
        parameters = {
            "repreprocess": True,
            "pca_ndim": 10,
            "basis": "X_pca",
        }
        trajectory_dict = cafe.method.cf_angle(self.fadata, prior_information, parameters)
        assert trajectory_dict.keys() == {"pseudotime"}

    def test_angle_new(self):
        parameters = {}
        trajectory_dict = cafe.method.cf_angle(self.fadata, **parameters)
        assert trajectory_dict.keys() == {"pseudotime"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

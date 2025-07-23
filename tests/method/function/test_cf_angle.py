import os

import pytest
import scanpy as sc

import cfe


class TestCFAngle:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_angle(self):
        # add priority and parameters
        prior_information = {}
        parameters = {
            "repreprocess": True,
            "pca_ndim": 10,
            "basis": "X_pca",
        }
        trajectory_dict = cfe.method.cf_angle(self.fadata, prior_information, parameters)
        assert trajectory_dict.keys() == {"pseudotime"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

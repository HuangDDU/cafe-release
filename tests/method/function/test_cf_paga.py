import pytest
import cfe

import os
import scanpy as sc


class TestCFPAGA():
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_paga(self):
        # add priority and parameeters
        prior_information = {
            "start_id": "cell1",
            "groups_id": self.fadata.obs["lineage"].tolist()
        }
        parameters = {"connectivity_cutoff": 0.8, "filter_features": False}
        trajectory_dict = cfe.method.cf_paga(self.fadata, prior_information, parameters)
        assert trajectory_dict.keys() == {"branch_network", "branches", "branch_progressions"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

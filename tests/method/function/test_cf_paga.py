import os

import pytest
import scanpy as sc

import cfe


class TestCFPAGA:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_paga(self):
        # add priority and parameeters
        prior_information = {"start_id": "cell1"}
        parameters = {"cluster_key": "lineage", "n_neighbors": 10, "n_dcs": 2, "connectivity_cutoff": 0.5}
        trajectory_dict = cfe.method.cf_paga(self.fadata, prior_information, parameters)
        assert trajectory_dict.keys() == {"branch_network", "branches", "branch_progressions"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

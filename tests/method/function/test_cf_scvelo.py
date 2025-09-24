import os

import pytest
import scanpy as sc

import cfe


class TestCFscVelo:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    @pytest.mark.skip(reason="TODO: fix")
    def test_scvelo(self):
        # add priority and parameeters
        prior_information = {
            "cluster_key": "lineage",
        }
        parameters = {}
        trajectory_dict = cfe.method.cf_scvelo(self.fadata, prior_information, parameters)
        assert trajectory_dict.keys() == {"milestone_network", "X_emb", "milestone_emb", "velocity_adata"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

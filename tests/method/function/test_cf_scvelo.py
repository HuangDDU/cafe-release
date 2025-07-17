import os

import pytest
import scanpy as sc

import cfe


class TestCFscVelo:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_scvelo(self):
        # add priority and parameeters
        prior_information = {
            "cluster_key": "lineage",
        }
        parameters = {}
        trajectory_dict = cfe.method.cf_scvelo(self.fadata, prior_information, parameters)
        assert trajectory_dict.keys() == {"velocity", "velocity_graph", "velocity_graph_neg", "neighbors", "obs_index", "var_index"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

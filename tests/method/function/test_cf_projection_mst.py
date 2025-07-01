import os

import pytest
import scanpy as sc

import cfe


class TestCFProjectionMST:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    @pytest.mark.skip(reason="TODO: fix")
    def test_projection_mst(self):
        # add priority and parameeters
        prior_information = {}
        parameters = {"ndim": 2, "distance_metric": "euclidean"}
        trajectory_dict = cfe.method.cf_projection_mst(self.fadata, prior_information, parameters)  # add parameters when inferring trajectory
        assert trajectory_dict.keys() == {"milestone_network", "X_emb", "milestone_emb"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

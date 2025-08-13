import os

import pytest
import scanpy as sc

import cfe


class TestCFProjectionMST:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_projection_mst_old(self):
        # add priority and parameeters
        prior_information = {}
        parameters = {
            "repreprocess": True,
            "pca_ndim": 10,
            "basis": "X_pca",
            "recluster": False,
            "cluster_key": "lineage",
            "distance_metric": "euclidean",
        }
        trajectory_dict = cfe.method.cf_projection_mst(self.fadata, prior_information, parameters)  # add parameters when inferring trajectory
        assert trajectory_dict.keys() == {"milestone_network", "X_emb", "milestone_emb"}

    def test_projection_mst_new(self):
        parameters = {
            "recluster": False,
            "cluster_key": "lineage",
        }
        trajectory_dict = cfe.method.cf_projection_mst(self.fadata, **parameters)
        assert trajectory_dict.keys() == {"milestone_network", "X_emb", "milestone_emb"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

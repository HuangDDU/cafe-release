import os

import pytest
import scanpy as sc

import cafe


class TestCFClusterMST:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cafe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_cluster_mst_old(self):
        # add priority and parameters
        prior_information = {}
        parameters = {
            "repreprocess": True,
            "pca_ndim": 10,
            "basis": "X_pca",
            "recluster": False,
            "cluster_key": "lineage",
            "distance_metric": "euclidean",
        }
        trajectory_dict = cafe.method.cf_cluster_mst(self.fadata, prior_information, parameters)  # add parameters when inferring trajectory
        assert trajectory_dict.keys() == {"milestone_network", "cluster"}

    def test_cluster_mst_new(self):
        parameters = {
            "recluster": False,
            "cluster_key": "lineage",
        }
        trajectory_dict = cafe.method.cf_cluster_mst(self.fadata, **parameters)
        assert trajectory_dict.keys() == {"milestone_network", "cluster"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

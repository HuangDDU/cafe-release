import os

import pytest
import scanpy as sc

import cfe


class TestCFClusterMST:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_cluster_mst(self):
        # add priority and parameeters
        prior_information = {"groups_id": self.fadata.obs["lineage"].tolist()}
        parameters = {"ndim": 2, "distance_metric": "euclidean"}
        trajectory_dict = cfe.method.cf_cluster_mst(self.fadata, prior_information, parameters)  # add parameters when inferring trajectory
        assert trajectory_dict.keys() == {"milestone_network", "cluster"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

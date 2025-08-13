import os

import pytest
import scanpy as sc

import cfe


class TestCFGraphMST:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_graph_mst_old(self):
        # add priority and parameeters
        prior_information = {}
        parameters = {"repreprocess": True, "pca_ndim": 2, "neighbors_kwargs": {}}
        trajectory_dict = cfe.method.cf_graph_mst(self.fadata, prior_information, parameters)
        assert trajectory_dict.keys() == {"cell_graph", "to_keep"}

    def test_graph_mst_new(self):
        parameters = {}
        trajectory_dict = cfe.method.cf_graph_mst(self.fadata, **parameters)
        assert trajectory_dict.keys() == {"cell_graph", "to_keep"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

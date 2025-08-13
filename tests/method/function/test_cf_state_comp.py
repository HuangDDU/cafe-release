import os

import pytest
import scanpy as sc

import cfe


class TestCFStateComp:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_state_comp_old(self):
        # add priority and parameeters
        prior_information = {}
        parameters = {"repreprocess": True, "n_comps": 2, "basis": "X_pca", "pseudotime_index": 1}
        trajectory_dict = cfe.method.cf_state_comp(self.fadata, prior_information, parameters)
        assert trajectory_dict.keys() == {"end_state_probabilities", "pseudotime"}

    def test_state_comp_new(self):
        parameters = {}
        trajectory_dict = cfe.method.cf_state_comp(self.fadata, **parameters)
        assert trajectory_dict.keys() == {"end_state_probabilities", "pseudotime"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

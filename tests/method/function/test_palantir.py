import os

import pytest
import scanpy as sc

import cfe


class TestCFPalantir:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cfe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    def test_palantir(self):
        pass
        # # add priority and parameeters
        # prior_information = {}
        # parameters = {"ndim": 2, "component": False}
        # trajectory_dict = cfe.method.cf_state_comp(self.fadata, prior_information, parameters)
        # assert trajectory_dict.keys() == {"end_state_probabilities", "pseudotime"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

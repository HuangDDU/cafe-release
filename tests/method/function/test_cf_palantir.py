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
        parameters = {
            "palantir_results_kwargs": {
                "early_cell": "cell1",
            }
        }
        trajectory_dict = cfe.method.cf_palantir(self.fadata, **parameters)
        assert "pseudotime" in trajectory_dict.keys()


if __name__ == "__main__":
    pytest.main(["-v", __file__])

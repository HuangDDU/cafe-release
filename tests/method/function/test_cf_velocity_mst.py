import os

import pytest
import scanpy as sc

import cafe


class TestCFVelocityMST:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")
        self.fadata = cafe.data.FateAnnData.from_anndata(adata)
        self.fadata.obs.index = self.fadata.obs["cell_id"].tolist()

    # TODO: issue: import packages from relative path
    # def test_velocity_mst(self):
    #     parameters = {}
    #     trajectory_dict = cafe.method.cf_velocity_mst(self.fadata, **parameters)
    #     assert "velocity" in trajectory_dict.keys()


if __name__ == "__main__":
    pytest.main(["-v", __file__])

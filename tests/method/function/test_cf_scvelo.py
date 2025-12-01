import os

import pytest
import scanpy as sc

import cafe

from .method_testcase import method_testcase


class TestCFscVelo:
    def setup_method(self):
        self.method_name = "scvelo"

        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/pancrease_scvelo_500_fadata.h5ad")
        fadata = cafe.data.FateAnnData.from_anndata(adata)

        self.adata = adata
        self.fadata = fadata
        self.parameters = {
            "repreprocess": True,
            "repreprocess_kwargs": {
                "min_shared_counts": 20,
                "n_top_genes": 2000,
                "n_pcs": 20,
                "n_neighbors": 10,
            },
            "velocity_kwargs": {},
            "velocity_graph_kwargs": {},
        }

    # Test raw trajectory dict
    def test_raw(self):
        # call function directly, use AnnData
        from cafe.method.function.cf_scvelo import scvelo

        trajectory_dict = scvelo(self.adata, **self.parameters)

        assert trajectory_dict.keys() == {
            "wrapper_type",
            "velocity",
            "velocity_graph",
            "velocity_graph_neg",
            "neighbors",
            "obs_index",
            "var_index",
        }  # check trajectory dict keys

    # Test three backends
    def test_function(self):
        fadata = method_testcase(self.adata, self.method_name, "python_function", self.parameters)
        assert fadata.is_wrapped_with_trajectory

    def test_conda(self):
        fadata = method_testcase(self.adata, self.method_name, "conda", self.parameters)
        assert fadata.is_wrapped_with_trajectory

    def test_docker(self):
        fadata = method_testcase(self.adata, self.method_name, "cafe_docker", self.parameters)
        assert fadata.is_wrapped_with_trajectory

    def test_call(self):
        cafe.method.cf_scvelo(self.fadata, self.parameters)
        assert self.fadata.is_wrapped_with_trajectory


if __name__ == "__main__":
    pytest.main(["-v", __file__])

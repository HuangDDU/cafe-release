import os
import sys

import pytest
import scanpy as sc

# don't import cfe here, because other conda environment may not have cfe relative package file

# Note: run the following command to run the test  in project dir with scvi-env conda environment
# conda activate scvi-env
# pytest -s --tb=long -m run_method test_cf_velovi.py

sys.path.append("../../../cfe/method/function")  # prepare relative package file for  file dir
sys.path.append("cfe/method/function")  # prepare relative package file for project dir


class TestCFVeloVI:
    def setup_method(self):
        self.adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../../data/bifurcating.h5ad")

    @pytest.mark.run_method
    def test_velovi(self):
        from cf_velovi import cf_velovi

        # add priority and parameters
        prior_information = {"cluster_key": "lineage"}
        parameters = {"max_epochs": 1}
        trajectory_dict = cf_velovi(self.adata, prior_information, parameters)
        assert trajectory_dict.keys() == {"velocity", "velocity_graph", "velocity_graph_neg", "neighbors", "obs_index", "var_index"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

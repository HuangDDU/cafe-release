import os

import pytest
import scanpy as sc

import cfe


def test_subsample():
    adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../data/bifurcating.h5ad")

    n_obs = 8

    # random subsample
    adata_subsample1 = cfe.preprocess.subsample(adata, n_obs)
    assert adata_subsample1.shape == (n_obs, adata.shape[1])

    # save specific cell
    save_cell_list = ["192", "278", "103"]
    adata_subsample2 = cfe.preprocess.subsample(adata, n_obs, save_cell_list=save_cell_list)
    assert len(list(set(adata_subsample2.obs.index) - set(save_cell_list))) == n_obs - len(save_cell_list)

    # save 1 cell for every cluster
    cluster_key = "lineage"
    adata_subsample3 = cfe.preprocess.subsample(adata, n_obs, save_cluster_key=cluster_key)
    assert set(adata_subsample3.obs[cluster_key]) == set(adata.obs[cluster_key])


if __name__ == "__main__":
    pytest.main(["-v", __file__])

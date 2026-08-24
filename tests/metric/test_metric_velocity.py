import numpy as np
import pytest
from scipy.sparse import csr_matrix

import cafe
from tests.metric.test_metric_cluster import (
    sample_fadata as fadata,  # import dataset as
)

fadata = fadata

CLUSTER_KEY = "clusters"
CLUSTER_LABELS = ["A", "A", "B", "B", "C", "C"]
BASIS = "X_umap"
EMBEDDING = np.array(
    [
        [0, 10],
        [8, 10],
        [12, 12],
        [20, 20],
        [15, 16],
        [22, 20],
    ]
)
CLUSTER_EDGES = [("A", "B"), ("B", "C")]


def _prepare_fadata(fadata, distances, n_neighbors):
    fadata.obs[CLUSTER_KEY] = CLUSTER_LABELS
    fadata.obsm[BASIS] = EMBEDDING
    fadata.uns["neighbors"] = {"params": {"n_neighbors": n_neighbors}}
    fadata.obsp["distances"] = csr_matrix(distances)


def test_calculate_velocity_metrics(fadata):
    distances = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    _prepare_fadata(fadata, distances, n_neighbors=2)

    metric_dict = cafe.metric.calculate_velocity_metrics(fadata, cluster_edges=CLUSTER_EDGES, cluster=CLUSTER_KEY, basis=BASIS)
    assert np.isfinite(metric_dict["velocity_cbdir"])
    assert np.isfinite(metric_dict["velocity_icvcoh"])


def test_calculate_velocity_metrics_with_variable_length_neighbors(fadata):
    # Each row has a different number of stored neighbors. The 16 non-zero
    # entries cannot be reshaped into rows of n_neighbors - 1 (3) entries.
    distances = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ]
    )
    _prepare_fadata(fadata, distances, n_neighbors=4)

    metric_dict = cafe.metric.calculate_velocity_metrics(fadata, cluster_edges=CLUSTER_EDGES, cluster=CLUSTER_KEY, basis=BASIS)

    assert 16 % (fadata.uns["neighbors"]["params"]["n_neighbors"] - 1) != 0  #
    assert np.isfinite(metric_dict["velocity_cbdir"])
    assert np.isfinite(metric_dict["velocity_icvcoh"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])

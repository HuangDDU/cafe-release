import numpy as np
import pytest
from scipy.sparse import csr_matrix

import cfe
from tests.metric.test_metric_cluster import (
    sample_fadata_ref as fadata,  # import dataset as
)

fadata = fadata


def test_calculate_velocity_metrics(fadata):
    # add cluster, embedding
    cluster = "clusters"
    fadata.obs[cluster] = ["A", "A", "B", "B", "B", "C"]
    basis = "X_umap"
    fadata.obsm[basis] = np.array(
        [
            [0, 10],
            [8, 10],
            [12, 12],
            [20, 20],
            [15, 16],
            [22, 20],
        ]
    )

    # add neighbor mannuly
    n_neighbors = 2
    fadata.uns["neighbors"] = {"params": {"n_neighbors": n_neighbors}}
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
    fadata.obsp["distances"] = csr_matrix(distances)

    # add velocity
    velocity = fadata.get_trajectory_pseudo_velocity(basis=basis)
    fadata.obsm[f"velocity_{basis.split('_')[1]}"] = velocity

    # calc metric
    cluster_edges = [("A", "B"), ("B", "C")]
    metric_dict = cfe.metric.calculate_velocity_metrics(fadata, cluster_edges=cluster_edges, cluster=cluster, basis=basis)
    assert metric_dict.keys() == {"CBDir", "ICVCoh"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])

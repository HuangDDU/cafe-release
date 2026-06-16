"""cosine_similarity strategy: direct velocity-to-cluster alignment.

Constructs a milestone network by computing cosine similarity between
each cluster's cell velocity vectors (in embedding space) and the
inter-cluster direction vectors between cluster centroids.

No PAGA dependency. Useful as a fallback when scVelo's PAGA pipeline
cannot be used.

Fixes the bug from the original monolithic implementation where
``self.obs[cluster]`` was used instead of ``adata.obs[cluster]``,
which could cause index misalignment when the temporary adata is
a subset (CellDancer) or synthetic (VeloAE).
"""

import numpy as np
import pandas as pd

from ..._logging import logger


def build_cosine_similarity(velo_input, threshold=0.2):
    """Build milestone network via cosine similarity between velocity and
    inter-cluster directions.

    For each pair of clusters (source → target), computes the mean cosine
    similarity between source cells' velocity vectors and the direction
    from source centroid to target centroid. Edges with similarity above
    ``threshold`` are kept.

    Parameters
    ----------
    velo_input : VelocityInput
        Standardized velocity data model.
    threshold : float
        Minimum cosine similarity to include an edge (default 0.2).

    Returns
    -------
    pd.DataFrame
        Milestone network with columns ``["from", "to", "length", "directed"]``.
    """
    adata = velo_input.adata
    velocity_embedding = velo_input.velocity_embedding
    milestone_emb = velo_input.milestone_emb

    # Determine cluster column
    cluster_col = milestone_emb.index.name
    if cluster_col is None:
        for col in adata.obs.columns:
            if adata.obs[col].dtype.name == "category":
                if set(adata.obs[col].cat.categories) == set(milestone_emb.index):
                    cluster_col = col
                    break
        if cluster_col is None:
            raise ValueError("Cannot determine cluster column for cosine similarity")

    # FIXED: use adata.obs instead of self.obs to avoid index misalignment
    cluster_list = adata.obs[cluster_col].cat.categories.to_list()
    cluster_connection_df = pd.DataFrame(0.0, index=cluster_list, columns=cluster_list)

    for source_cluster in cluster_list:
        # Get velocity vectors for cells in source cluster
        source_mask = (adata.obs[cluster_col] == source_cluster).values
        source_cell_velocity = velocity_embedding[np.where(source_mask)[0]]
        # Normalize
        source_cell_velocity = source_cell_velocity / (np.linalg.norm(source_cell_velocity, axis=1, keepdims=True) + 1e-6)

        for target_cluster in cluster_list:
            if source_cluster == target_cluster:
                continue
            cluster_direction = milestone_emb.loc[target_cluster].values - milestone_emb.loc[source_cluster].values
            cluster_direction = cluster_direction / (np.linalg.norm(cluster_direction) + 1e-6)
            # Mean cosine similarity between each cell's velocity and
            # the inter-cluster direction
            cosine_sims = (source_cell_velocity @ cluster_direction).mean()
            cluster_connection_df.loc[source_cluster, target_cluster] = cosine_sims

    logger.debug(f"cluster_connection_df:\n{cluster_connection_df.round(2)}")

    milestone_network = cluster_connection_df.stack().reset_index()
    milestone_network.columns = ["from", "to", "score"]
    milestone_network = milestone_network[milestone_network["score"] > threshold].copy()
    milestone_network["length"] = 1.0
    milestone_network["directed"] = True
    milestone_network = milestone_network[["from", "to", "length", "directed"]].reset_index(drop=True)

    logger.debug(f"cosine_similarity: built milestone_network with {len(milestone_network)} edges (threshold={threshold})")
    return milestone_network

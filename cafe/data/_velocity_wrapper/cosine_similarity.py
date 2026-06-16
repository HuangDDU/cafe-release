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


def build_cosine_similarity(velo_input, threshold=0.2, top_k=2, min_ratio=1.0):
    """Build milestone network via cosine similarity between velocity and
    inter-cluster directions.

    For each pair of clusters (source → target), computes the mean cosine
    similarity between source cells' velocity vectors and the direction
    from source centroid to target centroid.

    Three-stage filtering to produce clean, sparse networks:

    1. **absolute threshold**: score > ``threshold`` (default 0.2)
    2. **top-k per source**: keep at most ``top_k`` outgoing edges per cluster
       (default 2 — a cell population typically transitions to 1-2 fates)
    3. **forward/backward ratio**: forward score must exceed
       ``min_ratio`` × backward score (default 1.0, i.e. forward > backward)

    Parameters
    ----------
    velo_input : VelocityInput
        Standardized velocity data model.
    threshold : float
        Minimum cosine similarity to include an edge (default 0.2).
    top_k : int
        Maximum outgoing edges per source cluster. Set to 0 or None to disable.
    min_ratio : float
        Minimum ratio forward_score / backward_score. 1.0 means the forward
        direction must be stronger than the reverse. Set to 0 to disable.

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

    cluster_list = adata.obs[cluster_col].cat.categories.to_list()

    # --- compute all pairwise cosine scores ---
    scores = {}
    for source in cluster_list:
        source_mask = (adata.obs[cluster_col] == source).values
        src_velocity = velocity_embedding[np.where(source_mask)[0]]
        src_velocity = src_velocity / (np.linalg.norm(src_velocity, axis=1, keepdims=True) + 1e-6)

        for target in cluster_list:
            if source == target:
                continue
            direction = milestone_emb.loc[target].values - milestone_emb.loc[source].values
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            scores[(source, target)] = float((src_velocity @ direction).mean())

    # --- stage 1: absolute threshold ---
    candidates = {k: v for k, v in scores.items() if v > threshold}

    # --- stage 2: top-k per source ---
    if top_k:
        topk = {}
        for source in cluster_list:
            edges_from_src = [(k, v) for k, v in candidates.items() if k[0] == source]
            edges_from_src.sort(key=lambda x: x[1], reverse=True)
            for k, v in edges_from_src[:top_k]:
                topk[k] = v
        candidates = topk

    # --- stage 3: forward/backward ratio ---
    if min_ratio > 0:
        filtered = {}
        for (s, t), score in candidates.items():
            rev_score = scores.get((t, s), -1.0)
            if rev_score <= 0 or score >= min_ratio * rev_score:
                filtered[(s, t)] = score
        candidates = filtered

    # --- build milestone_network ---
    edges = []
    for (s, t), score in candidates.items():
        score = score  # todo: filter
        edges.append({"from": s, "to": t, "length": 1.0, "directed": True})

    milestone_network = pd.DataFrame(edges, columns=["from", "to", "length", "directed"])

    if len(milestone_network) == 0:
        logger.warning("cosine_similarity: no edges survived filtering " f"(threshold={threshold}, top_k={top_k}, min_ratio={min_ratio})")

    logger.debug(
        f"cosine_similarity: built milestone_network with {len(milestone_network)} edges "
        f"(threshold={threshold}, top_k={top_k}, min_ratio={min_ratio})"
    )
    return milestone_network

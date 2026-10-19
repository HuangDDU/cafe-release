"""Legacy cluster-flow graph and global categorical-time progress."""

import numpy as np
import pandas as pd
from scipy import sparse

from ..._logging import logger


def build_trajectory_base(
    fadata,
    tmaps: dict,
    time_key: str = None,
    cluster_key: str = None,
    flow_threshold: float = 0.1,
    relative_threshold: float = 0.3,
    normalize: bool = True,
    include_self_loop: bool = False,
):
    """Return the original cluster graph and ordinal-time cell positions.

    Matrix rows/columns follow fadata.obs order within each time point.
    Cells use their cluster's strongest outgoing transition, not cell-level
    fate probabilities. Progress uses category order, not actual time gaps.
    """
    # Get keys from prior_information if not specified
    if time_key is None:
        time_key = fadata.prior_information.get("time_key", "time")
    if cluster_key is None:
        cluster_key = fadata.prior_information.get("cluster", "clusters")

    obs = fadata.obs
    clusters = list(obs[cluster_key].cat.categories)
    n_clusters = len(clusters)
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}

    # ========== Step 1: Build cluster indicator matrices (for matrix multiplication) ==========
    def build_indicator_matrix(time_val):
        """Build sparse indicator matrix G_t (n_cells_t x n_clusters)"""
        mask = obs[time_key] == time_val
        cell_indices = np.where(mask.values)[0]
        cluster_codes = obs.loc[mask, cluster_key].map(cluster_to_idx).values
        n_cells = len(cell_indices)
        data = np.ones(n_cells, dtype=float)
        G = sparse.csr_matrix((data, (np.arange(n_cells), cluster_codes)), shape=(n_cells, n_clusters))
        return G

    # ========== Step 2: Aggregate cell-level Tmaps to cluster-level flow ==========
    cluster_flow = np.zeros((n_clusters, n_clusters))

    logger.debug(f"Aggregating {len(tmaps)} time-pair transport matrices...")
    for (t1, t2), tmap in tmaps.items():
        # Validate dimensions
        n_c1 = (obs[time_key] == t1).sum()
        n_c2 = (obs[time_key] == t2).sum()
        if tmap.shape != (n_c1, n_c2):
            logger.warning(f"Skipping {t1}->{t2}: Tmap shape {tmap.shape} != expected ({n_c1}, {n_c2})")
            continue

        # Build indicator matrices
        G1 = build_indicator_matrix(t1)
        G2 = build_indicator_matrix(t2)

        # Matrix multiplication: ClusterFlow = G1.T @ Tmap @ G2
        if sparse.issparse(tmap):
            flow = G1.T @ tmap @ G2
        else:
            flow = G1.T @ sparse.csr_matrix(tmap) @ G2
        cluster_flow += flow.toarray() if sparse.issparse(flow) else flow

    # Normalize by row
    if normalize:
        row_sums = cluster_flow.sum(axis=1, keepdims=True)
        cluster_flow = cluster_flow / (row_sums + 1e-10)

    cluster_flow_df = pd.DataFrame(cluster_flow, index=clusters, columns=clusters)

    # ========== Step 3: Build milestone_network from cluster flow ==========
    # Strategy: Use both absolute and relative thresholds to preserve bifurcations
    edges = []
    for source in clusters:
        outgoing = cluster_flow_df.loc[source].copy()

        # Optionally exclude self-loop
        if not include_self_loop:
            outgoing = outgoing.drop(source, errors="ignore")

        if len(outgoing) == 0 or outgoing.max() == 0:
            # No valid outgoing edges, add self-loop as fallback
            edges.append(
                {
                    "from": source,
                    "to": source,
                    "length": 1.0,
                    "directed": True,
                    "flow": cluster_flow_df.loc[source, source] if source in cluster_flow_df.columns else 0,
                }
            )
            continue

        # Compute dynamic threshold based on max flow
        max_flow = outgoing.max()
        dynamic_threshold = max(flow_threshold, relative_threshold * max_flow)

        # Filter edges by combined threshold
        valid_targets = outgoing[outgoing >= dynamic_threshold]

        if len(valid_targets) == 0:
            # Fallback: keep the strongest edge
            valid_targets = outgoing.nlargest(1)

        for target, flow in valid_targets.items():
            edges.append(
                {
                    "from": source,
                    "to": target,
                    "length": 1.0 / (flow + 1e-6),  # Higher flow → shorter length
                    "directed": True,
                    "flow": flow,
                }
            )

    if not edges:
        logger.warning("No edges found above flow_threshold. Consider lowering the threshold.")
        # Add self-loops as fallback
        for c in clusters:
            edges.append({"from": c, "to": c, "length": 1.0, "directed": True, "flow": 1.0})

    milestone_network = pd.DataFrame(edges)

    # ========== Step 4: Build progressions (assign cells to edges) ==========
    # Strategy: Assign each cell to the edge (source_cluster -> target_cluster)
    # where source_cluster is the cell's cluster, and target_cluster is chosen
    # based on the maximum outgoing flow. Percentage is based on time position.

    time_values = obs[time_key].cat.categories.tolist()
    time_to_norm = {t: i / max(len(time_values) - 1, 1) for i, t in enumerate(time_values)}

    progressions_list = []
    for cell_id in obs.index:
        cell_cluster = obs.loc[cell_id, cluster_key]
        cell_time = obs.loc[cell_id, time_key]

        # Find the best target cluster (highest flow from this cluster)
        outgoing = cluster_flow_df.loc[cell_cluster]
        # Exclude self-loop if there are other options
        if (outgoing.drop(cell_cluster, errors="ignore") > flow_threshold).any():
            target_cluster = outgoing.drop(cell_cluster, errors="ignore").idxmax()
        else:
            target_cluster = cell_cluster  # Self-loop

        # Percentage based on normalized time
        percentage = time_to_norm.get(cell_time, 0.5)

        progressions_list.append(
            {
                "cell_id": cell_id,
                "from": cell_cluster,
                "to": target_cluster,
                "percentage": percentage,
            }
        )

    progressions = pd.DataFrame(progressions_list)

    return {
        "milestone_network": milestone_network[["from", "to", "length", "directed"]],
        "progressions": progressions,
        "raw_wrapper_dict": {"cluster_flow": cluster_flow_df, "tmaps_keys": list(tmaps.keys())},
    }

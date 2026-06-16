"""scvelo_paga strategy: PAGA on full gene expression space.

Uses scVelo's PAGA implementation to derive milestone networks from
velocity-transition confidence on the full gene expression AnnData.

Requires ``velo_input.paga_ready == True`` (adata must have neighbors,
velocity_graph, and velocity_graph_neg pre-loaded).
"""

import scvelo as scv

from ..._logging import logger
from .base import _compute_edge_lengths


def build_scvelo_paga(velo_input, n_pcs=30, n_neighbors=30, use_embedding_distance=True):
    """Build milestone network via PAGA on full gene expression space.

    Parameters
    ----------
    velo_input : VelocityInput
        Standardized velocity data model. Must have ``paga_ready == True``.
    n_pcs : int
        Number of PCs for scvelo moments (used only if paga_ready is False).
    n_neighbors : int
        Number of neighbors for scvelo moments (used only if paga_ready is False).
    use_embedding_distance : bool
        If True, compute edge lengths from actual embedding distances instead
        of hardcoding ``length = 1``.

    Returns
    -------
    pd.DataFrame
        Milestone network with columns ``["from", "to", "length", "directed"]``.
    """
    adata = velo_input.adata
    # PAGA requires the cluster column as a categorical in obs
    cluster_col = velo_input.milestone_emb.index.name
    if cluster_col is None:
        # Infer cluster column from the milestone_emb categories
        for col in adata.obs.columns:
            if adata.obs[col].dtype.name == "category":
                if set(adata.obs[col].cat.categories) == set(velo_input.milestone_emb.index):
                    cluster_col = col
                    break
        if cluster_col is None:
            raise ValueError("Cannot determine cluster column for PAGA")

    # If paga is not ready, recompute
    if not velo_input.paga_ready:
        logger.debug("paga_ready=False; recomputing moments and velocity_graph")
        scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
        scv.tl.velocity_graph(adata)

    scv.tl.paga(adata, groups=cluster_col)
    df = scv.get_df(adata, "paga/transitions_confidence", precision=2).T

    milestone_network = (
        df.reset_index().rename(columns={"index": "from"}).melt(id_vars="from", var_name="to", value_name="length").query("`length` > 0")
    )

    if use_embedding_distance:
        milestone_network["length"] = _compute_edge_lengths(milestone_network, velo_input.milestone_emb)
    else:
        milestone_network["length"] = 1.0

    milestone_network["directed"] = True
    milestone_network = milestone_network.reset_index(drop=True)

    logger.debug(f"scvelo_paga: built milestone_network with {len(milestone_network)} edges")
    return milestone_network

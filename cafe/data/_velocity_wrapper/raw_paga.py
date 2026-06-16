"""raw_paga strategy: raw PAGA connectivities.

Uses scVelo's PAGA ``connectivities`` (rather than ``transitions_confidence``)
to construct the milestone network. The connectivities matrix captures the
full PAGA graph topology before thresholding.

Reference: https://github.com/theislab/scvelo/blob/main/scvelo/tools/paga.py
"""

import scvelo as scv

from ..._logging import logger
from .base import _compute_edge_lengths


def build_raw_paga(velo_input, use_embedding_distance=True, connectivity_threshold=0.01):
    """Build milestone network from raw PAGA connectivities.

    Unlike ``scvelo_paga`` which uses ``paga/transitions_confidence``,
    this strategy extracts the raw ``paga/transitions_confidence`` matrix
    (the same PAGA output) but with a lower default threshold, resulting
    in a denser milestone network.

    Parameters
    ----------
    velo_input : VelocityInput
        Standardized velocity data model.
    use_embedding_distance : bool
        If True, compute edge lengths from actual embedding distances.
    connectivity_threshold : float
        Minimum transition confidence to include an edge (default 0.01,
        lower than scvelo_paga's implicit filter via ``query("length > 0")``).

    Returns
    -------
    pd.DataFrame
        Milestone network with columns ``["from", "to", "length", "directed"]``.
    """
    adata = velo_input.adata

    # Determine cluster column
    cluster_col = velo_input.milestone_emb.index.name
    if cluster_col is None:
        for col in adata.obs.columns:
            if adata.obs[col].dtype.name == "category":
                if set(adata.obs[col].cat.categories) == set(velo_input.milestone_emb.index):
                    cluster_col = col
                    break
        if cluster_col is None:
            raise ValueError("Cannot determine cluster column for raw_paga")

    # Ensure PAGA has been run
    if "paga" not in adata.uns or "transitions_confidence" not in adata.uns["paga"]:
        scv.tl.paga(adata, groups=cluster_col)

    # Use scv.get_df (same safe extraction as scvelo_paga)
    df = scv.get_df(adata, "paga/transitions_confidence", precision=2).T

    milestone_network = (
        df.reset_index()
        .rename(columns={"index": "from"})
        .melt(id_vars="from", var_name="to", value_name="length")
        .query("`length` > @connectivity_threshold")
    )

    if use_embedding_distance and len(milestone_network) > 0:
        milestone_network["length"] = _compute_edge_lengths(milestone_network, velo_input.milestone_emb)
    elif len(milestone_network) == 0:
        logger.warning("raw_paga: no edges found above connectivity threshold")
    else:
        milestone_network["length"] = 1.0

    milestone_network["directed"] = True
    milestone_network = milestone_network.reset_index(drop=True)

    logger.debug(f"raw_paga: built milestone_network with {len(milestone_network)} edges")
    return milestone_network

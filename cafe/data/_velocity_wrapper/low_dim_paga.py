"""low_dim_paga strategy: PAGA on low-dimensional embedding space.

Used when the velocity method provides a pre-computed low-dimensional
velocity embedding (e.g. CellDancer, Dynamo). Constructs a synthetic
AnnData from the embedding space and recomputes neighbors, velocity_graph,
and PAGA on this low-dimensional representation.

Fixes the bug from the original monolithic implementation where
``scv.get_df(adata, ...)`` was called instead of ``scv.get_df(new_adata, ...)``.
"""

import scanpy as sc
import scvelo as scv

from ..._logging import logger
from .base import _compute_edge_lengths


def build_low_dim_paga(velo_input, n_neighbors=15, use_embedding_distance=True):
    """Build milestone network via PAGA on low-dimensional embedding space.

    Constructs a synthetic AnnData where both "spliced" and "unspliced"
    layers are set to the expression embedding, and "velocity" is set to
    the low-dimensional velocity_embedding. Then recomputes neighbors,
    velocity_graph, and PAGA on this space.

    Parameters
    ----------
    velo_input : VelocityInput
        Standardized velocity data model.
    n_neighbors : int
        Number of neighbors for the low-dim neighbor graph.
    use_embedding_distance : bool
        If True, compute edge lengths from actual embedding distances.

    Returns
    -------
    pd.DataFrame
        Milestone network with columns ``["from", "to", "length", "directed"]``.
    """
    adata = velo_input.adata

    # Determine the embedding key from velo_input
    basis = velo_input.velocity_basis.replace("velocity_", "X_")

    # Construct synthetic AnnData on the embedding space
    new_adata = sc.AnnData(
        X=adata.obsm[basis],
        obs=adata.obs,
        obsm=adata.obsm,
        obsp=adata.obsp,
        uns=adata.uns,
    )
    new_adata.layers["spliced"] = adata.obsm[basis]
    new_adata.layers["unspliced"] = adata.obsm[basis]
    new_adata.layers["velocity"] = velo_input.velocity_embedding

    # Recompute velocity graph on low-dim embedding
    sc.pp.neighbors(new_adata, n_neighbors=n_neighbors)
    scv.tl.velocity_graph(new_adata, show_progress_bar=False)
    scv.tl.paga(new_adata, groups=velo_input.milestone_emb.index.name)

    # FIXED: use new_adata instead of adata (original bug used adata here)
    df = scv.get_df(new_adata, "paga/transitions_confidence", precision=2).T

    milestone_network = (
        df.reset_index().rename(columns={"index": "from"}).melt(id_vars="from", var_name="to", value_name="length").query("`length` > 0")
    )

    if use_embedding_distance:
        milestone_network["length"] = _compute_edge_lengths(milestone_network, velo_input.milestone_emb)
    else:
        milestone_network["length"] = 1.0

    milestone_network["directed"] = True
    milestone_network = milestone_network.reset_index(drop=True)

    logger.debug(f"low_dim_paga: built milestone_network with {len(milestone_network)} edges")
    return milestone_network

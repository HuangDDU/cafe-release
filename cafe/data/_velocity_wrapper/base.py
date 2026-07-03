"""Core data model and dispatcher for velocity-based trajectory construction.

This module provides the standardized VelocityInput data model and helper
functions that extract and normalize velocity information from different
RNA velocity method outputs into a common format for milestone network
construction.
"""

from collections import namedtuple

import anndata as ad
import pandas as pd
import scvelo as scv

from ..._logging import logger

# ---------------------------------------------------------------------------
# Standardized data model
# Use a namedtuple instead of a class to define a clear, immutable data structure for velocity inputs.
# ---------------------------------------------------------------------------

VelocityInput = namedtuple(
    "VelocityInput",
    [
        "adata",  # ad.AnnData: constructed AnnData (scvelo-compatible)
        "velocity_embedding",  # np.ndarray: low-dim velocity embedding (n_cells x n_dims)
        "velocity_basis",  # str: key name, e.g. "velocity_umap"
        "X_emb",  # pd.DataFrame: cell embedding for projection (index = cell IDs)
        "milestone_emb",  # pd.DataFrame: cluster centroid embeddings (index = cluster names)
        "paga_ready",  # bool: whether adata has neighbors + velocity_graph + velocity_graph_neg
    ],
)

# ---------------------------------------------------------------------------
# Strategy registry (populated lazily to avoid circular imports)
# ---------------------------------------------------------------------------

VELOCITY_STRATEGIES = {}


def _register_strategies():
    """Lazily register strategy builders to avoid circular imports."""
    if VELOCITY_STRATEGIES:
        return
    from .cosine_similarity import build_cosine_similarity
    from .low_dim_paga import build_low_dim_paga
    from .raw_paga import build_raw_paga
    from .scvelo_paga import build_scvelo_paga

    VELOCITY_STRATEGIES.update(
        {
            "scvelo_paga": build_scvelo_paga,
            "low_dim_paga": build_low_dim_paga,  # for celldancer
            "raw_paga": build_raw_paga,  # TODO: implement raw_paga strategy refer to scv.tl.paga
            "cosine_similarity": build_cosine_similarity,
        }
    )


def choose_or_check_strategy(trajectory_dict: dict, milestone_network_strategy: str):
    velocity = trajectory_dict.get("velocity")
    velocity_embedding = trajectory_dict.get("velocity_embedding")
    if milestone_network_strategy == "auto":
        # TODO: milestone network strategy choice
        if velocity_embedding is not None and velocity is None:
            milestone_network_strategy = "low_dim_paga"
        else:
            milestone_network_strategy = "scvelo_paga"
        logger.debug(f"Auto-selected milestone network strategy: '{milestone_network_strategy}'")
    else:
        # TODO: check strategy choice
        valid = True
        if not valid:
            logger.warning("Selected milestone network strategy is not valid.")
            milestone_network_strategy = "scvelo_paga"
    return milestone_network_strategy


# ---------------------------------------------------------------------------
# AnnData construction
# ---------------------------------------------------------------------------


def prepare_anndata_for_velocity(
    fadata,
    trajectory_dict: dict,
    cluster: str,
    basis: str,
) -> tuple:
    """Construct a standard scvelo-compatible AnnData from a trajectory_dict.

    Handles the three AnnData construction branches:
    - VeloAE:  ``X is not None`` → build from latent space matrix
    - CellDancer/Dynamo: ``obs_index`` / ``var_index`` subset → subset of self
    - Standard: full ``self.to_anndata()``

    Parameters
    ----------
    fadata : FateAnnData
        The FateAnnData instance.
    trajectory_dict : dict
        Velocity method output dict. Expected keys: X, obs_index, var_index,
        velocity_embedding.
    cluster : str
        Cluster column name in ``.obs``.
    basis : str
        Embedding key in ``.obsm`` (e.g. ``"X_umap"``).

    Returns
    -------
    adata : ad.AnnData
        The constructed AnnData ready for velocity computation.
    force_strategy : str or None
        ``"low_dim_paga"`` if velocity_embedding is pre-computed, else None.
    """
    X = trajectory_dict.get("X")
    obs_index = trajectory_dict.get("obs_index")
    var_index = trajectory_dict.get("var_index")
    # velocity_embedding = trajectory_dict.get("velocity_embedding")

    # Branch 1: VeloAE — reconstruct AnnData from latent space
    if X is not None:
        adata = ad.AnnData(X)
        adata.obs.index = obs_index if obs_index is not None else fadata.obs.index
        adata.var.index = var_index if var_index is not None else fadata.var.index
        # Align cluster labels and basis embedding from the original fadata
        adata.obs[cluster] = fadata[adata.obs.index].obs[cluster]
        adata.obsm[basis] = fadata[adata.obs.index].obsm[basis]
        logger.debug(f"VeloAE branch: constructed adata with shape {adata.shape}")
    # Branch 2: CellDancer/Dynamo — subset to cells with valid velocity
    elif (obs_index is not None) or (var_index is not None):
        obs_index = fadata.obs.index if obs_index is None else obs_index
        var_index = fadata.var.index if var_index is None else var_index
        adata = fadata[obs_index, var_index].to_anndata()
        logger.debug(f"Subset branch: filtered adata with shape {adata.shape}")
    # Branch 3: Standard — full copy
    else:
        adata = fadata.to_anndata()
        logger.debug(f"Standard branch: copied adata with shape {adata.shape}")

    return adata


# ---------------------------------------------------------------------------
# Velocity embedding computation
# ---------------------------------------------------------------------------


def compute_velocity_embedding(
    adata: ad.AnnData,
    trajectory_dict: dict,
    basis: str,
    n_pcs: int = 30,
    n_neighbors: int = 30,
) -> tuple:
    """Compute or extract low-dimensional velocity embedding.

    Three paths:
    1. Pre-computed ``velocity_embedding`` in trajectory_dict → return directly
    2. Pre-computed ``velocity_graph`` + ``velocity_graph_neg`` + neighbors
       → inject into adata, call ``scv.tl.velocity_embedding``
    3. Neither → recompute via ``scv.pp.moments`` + ``scv.tl.velocity_graph``
       + ``scv.tl.velocity_embedding``

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData constructed by ``prepare_anndata_for_velocity``.
    trajectory_dict : dict
        Velocity method output dict.
    basis : str
        Embedding key (e.g. ``"X_umap"``).
    n_pcs : int
        Number of PCs for ``scv.pp.moments`` (only used in recompute path).
    n_neighbors : int
        Number of neighbors for ``scv.pp.moments`` (only used in recompute path).

    Returns
    -------
    velocity_embedding : np.ndarray
        Low-dimensional velocity embedding (n_cells x n_dims).
    velocity_basis : str
        Key name (e.g. ``"velocity_umap"``).
    """
    velocity_basis = f"velocity_{basis[2:]}"  # strip "X_" prefix

    # Path 1: pre-computed velocity_embedding (CellDancer / Dynamo)
    velocity_embedding = trajectory_dict.get("velocity_embedding")
    if velocity_embedding is not None:
        logger.debug("Using pre-computed velocity_embedding")
        return velocity_embedding, velocity_basis

    # Set up the velocity layer
    velocity = trajectory_dict.get("velocity")
    if velocity is not None:
        adata.layers["velocity"] = velocity

    velocity_graph = trajectory_dict.get("velocity_graph")
    velocity_graph_neg = trajectory_dict.get("velocity_graph_neg")
    neighbors = trajectory_dict.get("neighbors")

    # Path 2: pre-computed velocity_graph + velocity_graph_neg + neighbors
    if (velocity_graph is not None) and (velocity_graph_neg is not None) and (neighbors is not None):
        adata.uns["velocity_graph"] = velocity_graph
        adata.uns["velocity_graph_neg"] = velocity_graph_neg
        adata.uns["neighbors"] = {}
        adata.obsp["distances"] = neighbors["distances"]
        adata.obsp["connectivities"] = neighbors["connectivities"]
        logger.debug("Injected pre-computed velocity_graph, velocity_graph_neg, and neighbors")
    else:
        # Path 3: recompute everything
        logger.debug("Recomputing moments, velocity_graph (n_pcs=%d, n_neighbors=%d)", n_pcs, n_neighbors)
        scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
        scv.tl.velocity_graph(adata)

    logger.debug("Computing velocity_embedding via scvelo")
    scv.tl.velocity_embedding(adata, basis=basis[2:])
    velocity_embedding = adata.obsm[velocity_basis]

    return velocity_embedding, velocity_basis


# ---------------------------------------------------------------------------
# Milestone embedding computation
# ---------------------------------------------------------------------------


def compute_milestone_embeddings(
    adata: ad.AnnData,
    cluster: str,
    basis: str,
) -> pd.DataFrame:
    """Compute cluster-centroid embeddings from the AnnData embedding.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with ``.obsm[basis]`` and ``.obs[cluster]``.
    cluster : str
        Cluster column name in ``.obs``.
    basis : str
        Embedding key in ``.obsm``.

    Returns
    -------
    pd.DataFrame
        Cluster centroid coordinates, indexed by cluster category names.
    """
    X_emb = pd.DataFrame(adata.obsm[basis], index=adata.obs.index)
    milestone_emb = adata.obs.groupby(cluster).apply(lambda x: X_emb.loc[x.index].mean(axis=0))
    milestone_emb.index = list(adata.obs[cluster].cat.categories)
    return milestone_emb


# ---------------------------------------------------------------------------
# Edge length computation
# ---------------------------------------------------------------------------


def _compute_edge_lengths(
    milestone_network: pd.DataFrame,
    milestone_emb: pd.DataFrame,
    min_length: float = 0.01,
) -> pd.Series:
    """Compute edge lengths as Euclidean distances between milestone centroids.

    Replaces the old hardcoded ``length = 1`` with actual embedding-space
    distances, following the same pattern as ``add_trajectory_mannually``.

    Parameters
    ----------
    milestone_network : pd.DataFrame
        DataFrame with ``"from"`` and ``"to"`` columns.
    milestone_emb : pd.DataFrame
        Cluster centroid coordinates, indexed by cluster name.
    min_length : float
        Minimum edge length to prevent zero-length edges.

    Returns
    -------
    pd.Series
        Edge lengths clipped to at least ``min_length``.
    """
    from sklearn.metrics.pairwise import pairwise_distances

    dis = pd.DataFrame(
        pairwise_distances(milestone_emb, metric="euclidean"),
        index=milestone_emb.index,
        columns=milestone_emb.index,
    )
    lengths = milestone_network.apply(lambda row: dis.loc[row["from"], row["to"]], axis=1)
    return lengths.clip(lower=min_length)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def build_milestone_network(
    velo_input: VelocityInput,
    strategy: str = "scvelo_paga",
    strategy_kwargs: dict = None,
) -> pd.DataFrame:
    """Build a milestone network from velocity data using the given strategy.

    Parameters
    ----------
    velo_input : VelocityInput
        Standardized velocity data model.
    strategy : str
        Strategy name. One of ``"scvelo_paga"``, ``"low_dim_paga"``,
        ``"raw_paga"``, or ``"cosine_similarity"``.
    strategy_kwargs : dict, optional
        Additional keyword arguments passed to the strategy builder.

    Returns
    -------
    pd.DataFrame
        Milestone network with columns ``["from", "to", "length", "directed"]``.

    Raises
    ------
    ValueError
        If the strategy name is unknown.
    """
    _register_strategies()

    if strategy not in VELOCITY_STRATEGIES:
        raise ValueError(f"Unknown velocity milestone network strategy: '{strategy}'. " f"Available: {list(VELOCITY_STRATEGIES.keys())}")

    kwargs = strategy_kwargs or {}
    return VELOCITY_STRATEGIES[strategy](velo_input, **kwargs)

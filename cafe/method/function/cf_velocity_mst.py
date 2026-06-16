from typing import Optional

import anndata as ad
import numpy as np

try:
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    from cafe.method.function.method_decorator import method_info
    from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="velocity_mst",
    version="0.0.1",
    description="VelocityMST: Minimum Spanning Tree baseline for velocity wrapper.",
    wrapper_type="velocity",
    doi="",
    github_url="",
    use_gpu=False,
    cpu_parallelization=True,
    available=True,
)
def velocity_mst(
    adata: ad.AnnData,
    start_cell: str,
    cluster: str = "clusters",
    basis: str = "X_pca",
    repreprocess: bool = True,
    repreprocess_kwargs: Optional[dict] = None,
    distance_metric: str = "euclidean",
):
    """VelocityMST: MST baseline for velocity wrapper.

    Reuses ``cluster_mst`` to build a directed MST, then derives velocity
    vectors from the directed edge structure and recovers high-dimensional
    gene-space velocity via PCA inverse transform.
    """
    if repreprocess_kwargs is None:
        repreprocess_kwargs = {}

    # 1. Preprocess — ensure PCA and neighbors exist
    if repreprocess:
        preprocess_pipeline(adata, style="scvelo", **repreprocess_kwargs)

    # 2. Build directed MST via cluster_mst
    # cafe docker backend is not available  here because of import function in relative path
    from cafe.method.function.cf_cluster_mst import cluster_mst

    mst_result = cluster_mst(
        adata=adata,
        start_cell=start_cell,
        cluster=cluster,
        basis=basis,
        repreprocess=False,  # already done above
        distance_metric=distance_metric,
    )
    milestone_network = mst_result["milestone_network"]
    cluster_labels = mst_result["cluster"]  # pd.Series, index aligned to adata
    centers = mst_result["centers"]  # pd.DataFrame, index=cluster_id, columns=embedding dims
    milestone_ids = centers.index.tolist()

    # 4. Compute velocity_embedding — cluster-level direction vectors
    X_emb = adata.obsm[basis]
    n_dims = X_emb.shape[1]
    cluster_direction = {}

    # same as pseudo-velocity
    for cid in milestone_ids:
        outgoing = milestone_network[milestone_network["from"] == cid]
        if len(outgoing) > 0:
            total_vec = np.zeros(n_dims)
            total_w = 0.0
            for _, edge in outgoing.iterrows():
                target = edge["to"]
                direction = centers.loc[target].values - centers.loc[cid].values
                w = 1.0 / max(edge["length"], 1e-6)
                total_vec += direction * w
                total_w += w
            cluster_direction[cid] = total_vec / max(total_w, 1e-6)
        else:
            parent = milestone_network[milestone_network["to"] == cid]
            if len(parent) > 0:
                p = parent.iloc[0]["from"]
                cluster_direction[cid] = centers.loc[cid].values - centers.loc[p].values
            else:
                cluster_direction[cid] = np.zeros(n_dims)

    velocity_embedding = np.array([cluster_direction[c] for c in cluster_labels], dtype=np.float64)

    # 5. Recover high-dimensional velocity via PCA inverse transform
    if basis == "X_pca" and "PCs" in adata.varm:
        # recovery high-dim velocity via PCA inverse transform
        velocity = np.asarray(velocity_embedding @ adata.varm["PCs"].T, dtype=np.float64)
        adata.layers["velocity"] = velocity
        milestone_network_strategy = "scvelo_paga"
    else:
        # low-dim velocity
        velocity = None
        milestone_network_strategy = "low_dim_paga"

    # 6. Neighbors
    neighbors = None
    if "distances" in adata.obsp:
        neighbors = {
            "distances": adata.obsp["distances"],
            "connectivities": adata.obsp["connectivities"],
        }

    print("milestone_network_strategy:", milestone_network_strategy)
    trajectory_dict = {
        "wrapper_type": "velocity",
        "milestone_network_strategy": milestone_network_strategy,
        "velocity": velocity,
        "velocity_graph": None,
        "velocity_graph_neg": None,
        "velocity_embedding": velocity_embedding,
        "neighbors": neighbors,
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,
        "basis": basis,  # is still available for velocity wrapper
    }

    return trajectory_dict

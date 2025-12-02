from typing import Literal, Optional

import anndata as ad
import networkx as nx
import pandas as pd
import scanpy as sc
from sklearn.metrics.pairwise import pairwise_distances

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cafe environment
    from cafe.method.function.method_decorator import method_info
    from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="cluster_mst",
    version="0.0.1",
    description="Cluster MST: baseline for cluster wrapper, creating a Minimum Spanning Tree (MST) on cluster centers.",
    wrapper_type="cluster",
)
def cluster_mst(
    adata: ad.AnnData,
    repreprocess: bool = True,
    basis: str = "X_pca",
    recluster: bool = True,
    cluster: str = "clusters",
    distance_metric: Optional[Literal["euclidean", "cosine", "manhattan", "cityblock", "l1", "l2"]] = "euclidean",
):
    """Cluster MST: baseline for cluster wrapper, creating a Minimum Spanning Tree (MST) on cluster centers.

    This method first clusters the cells (or uses existing clusters), calculates the
    center of each cluster in a given embedding, and then constructs a Minimum
    Spanning Tree (MST) connecting these centers to represent the trajectory backbone.

    Args:
        adata (ad.AnnData): The input AnnData object.
        repreprocess (bool, optional): Whether to run the preprocessing pipeline.
        basis (str, optional): The embedding in `.obsm` to use for calculating cluster centers.
        recluster (bool, optional): If True, re-computes cell clusters using the Leiden algorithm.
        cluster (str, optional): The key in `adata.obs` where cluster information is stored/saved.
        distance_metric (str, optional): The distance metric to use for calculating distances between cluster centers.

    Returns:
        dict: A trajectory dict compatible with the 'lineage' wrapper, containing the milestone network and cluster assignments.
    """
    # 1. preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="scanpy", if_neighbors=True)  # ensure neighbors are computed
    adata.obs.reset_index(drop=True, inplace=True)  # for X_emb index consistency
    X_emb = adata.obsm[basis]

    # 2. execute method
    # (1) Cluster cells, with the center point as a milestone
    if recluster:
        # new cluster
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)
        cluster = "leiden"
    # (2) Calculate the low dimensional coordinates of the clustering centers
    centers = adata.obs.groupby(cluster).apply(lambda x: X_emb[list(x.index)].mean(axis=0))
    centers = pd.DataFrame(centers.tolist(), index=centers.index)
    milestone_ids = centers.index.tolist()
    cluster_milestones = adata.obs[cluster]
    # (3) Calculate the distance between cluster centers
    dis = pd.DataFrame(pairwise_distances(centers, metric=distance_metric), index=milestone_ids, columns=milestone_ids)
    dis_df = pd.DataFrame(data=dis.unstack().reset_index().values, columns=["from", "to", "weight"])  # width data to long data
    # (4) Calculate the distance between milestones and construct the minimum spanning tree as the milestone network
    G = nx.from_pandas_edgelist(dis_df, source="from", target="to", edge_attr="weight")
    mst = nx.minimum_spanning_tree(G, weight="weight")

    # 3. extract results
    milestone_network = nx.to_pandas_edgelist(mst)
    milestone_network.rename(columns={"source": "from", "target": "to", "weight": "length"}, inplace=True)
    milestone_network["directed"] = False

    # 4. save results
    trajectory_dict = {
        "wrapper_type": "cluster",
        "milestone_network": milestone_network,
        "cluster": cluster_milestones,
    }

    return trajectory_dict

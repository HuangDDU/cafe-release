# refer to: https://github.com/qiaochen/VeloAE/blob/main/veloproj/eval_util.py
"""Evaluation utility functions.

This module contains util functions for computing evaluation scores.

"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .._logging import logger


def summary_scores(all_scores):
    """Summarize group scores.

    Args:
        all_scores (dict{str,list}): {group name: score list of individual cells}.

    Returns:
        dict{str,float}: Group-wise aggregation scores.
        float: score aggregated on all samples

    """
    sep_scores = {k: np.mean(s) for k, s in all_scores.items() if s}
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg


def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type

    Args:
        adata (Anndata): Anndata object.
        nodes (list): Indexes for cells
        target (str): Cluster name.
        k_cluster (str): Cluster key in adata.obs dataframe

    Returns:
        list: Selected cells.

    """
    return nodes[adata.obs[k_cluster][nodes].values == target]


def cross_boundary_correctness(adata, k_cluster, k_velocity, cluster_edges, return_raw=False, x_emb="X_umap"):
    """Cross-Boundary Direction Correctness Score (A->B)

    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        x_emb (str): key to x embedding for visualization.

    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.

    """
    scores = {}
    all_scores = {}

    v_emb = adata.obsm[k_velocity]  # velocity embedding space
    x_emb = adata.obsm[x_emb]  # expression embedding space

    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns["neighbors"]["indices"][sel]  # [n * 30] # TODO: update here mannuly add indices

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))

        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])  # here use mean


def inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=False):
    """In-cluster Coherence Score.

    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        return_raw (bool): return aggregated or raw scores.

    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.

    """
    velocities = adata.obsm[k_velocity]

    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}
    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns["neighbors"]["indices"][sel]
        same_cat_nodes = map(lambda nodes: keep_type(adata, nodes, cat, k_cluster), nbs)
        # velocities = adata.layers[k_velocity] # replace by
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean() for ith, nodes in enumerate(same_cat_nodes) if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])


def calculate_velocity_metrics(adata, cluster_edges, cluster, basis="X_umap", return_raw=True, summary=True):
    """Evaluate velocity estimation results using 5 metrics.

    Args:
        adata (Anndata): Anndata object.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        cluster (str): key to the cluster column in adata.obs DataFrame.
        basis (str): key to x embedding for visualization.
        return_raw (bool): return aggregated or raw scores.
        summary (bool): if not return_raw, whether to return summary scores.


    Returns:
        dict: aggregated metric scores.

    """
    k_velocity = f"velocity_{basis.split('_')[1]}"

    neighbor_dict = adata.uns["neighbors"]
    if "indices" not in neighbor_dict:
        logger.debug("extract knn indices to 'adata.uns['neighbors']['indices']' for metric calculation")
        n_neighbors = neighbor_dict["params"]["n_neighbors"]
        distances = adata.obsp["distances"]  # csr matrix
        neighbor_dict["indices"] = distances.indices.reshape(-1, n_neighbors - 1)

    crs_bdr_crc = cross_boundary_correctness(adata, cluster, k_velocity, cluster_edges, return_raw, basis)
    ic_coh = inner_cluster_coh(adata, cluster, k_velocity, return_raw)

    if return_raw:
        # if return raw scores, do nothing
        if summary:
            logger.debug("'return_raw`'and 'summary' both set True, only 'return_raw' is effective.")
    else:
        if summary:
            # if don't return raw, just summary score, do it
            crs_bdr_crc = crs_bdr_crc[1]
            ic_coh = ic_coh[1]
    return {
        "CBDir": crs_bdr_crc,
        "ICVCoh": ic_coh,
    }

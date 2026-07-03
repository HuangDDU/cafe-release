import numpy as np
from scipy import sparse

from .._logging import logger


def _pearson_corr_with_vector(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate Pearson correlation between each column in X and y."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)

    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 2:
        return np.zeros(X.shape[1], dtype=float)

    y = y[valid_mask]
    X = X[valid_mask]

    y_centered = y - y.mean()
    y_std = y_centered.std()
    if y_std == 0:
        return np.zeros(X.shape[1], dtype=float)

    X_centered = X - X.mean(axis=0)
    X_std = X_centered.std(axis=0)

    cov = (X_centered * y_centered[:, None]).mean(axis=0)
    denom = X_std * y_std

    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    return corr


def find_edge_feature_gene(
    fadata,
    edge_list: list,
    model_name: str = None,
    top_n: int = None,
):
    """Find top feature genes for each edge by pseudotime-expression correlation.

    For each edge, this function:
    1) subsets cells on the edge,
    2) computes pseudotime from edge start milestone,
    3) calculates Pearson correlation between each gene and pseudotime,
    4) ranks genes by absolute correlation and returns top-N gene names.
    """
    edge_dict = {}  # edge: feature gene list (sorted)

    n_var = fadata.shape[1]
    if top_n is None:
        top_n = n_var
        logger.debug(f"parameter `top_n` is None, save all genes ({top_n})")
    else:
        if top_n > n_var:
            top_n = n_var
            logger.warning(f"parameter `top_n` is too large, save all genes ({top_n})")

    for edge in edge_list:
        edge_key = tuple(edge)

        # extract fadata by edge
        fadata_sub = fadata.subset_trajectory(edge_list=[edge_key], model_name=model_name)

        # calculate pseudotime based on milestone network
        pseudotime = fadata_sub.get_trajectory_pseudotime(start_milestone=edge_key[0], model_name=model_name)  # (n_obs, )

        # sort gene by expression correlation with pseudotime
        X = fadata_sub.X  # (n_obs, n_var)
        if sparse.issparse(X):
            X = X.toarray()

        corr = _pearson_corr_with_vector(X=X, y=np.asarray(pseudotime))  # (n_var,)
        rank_idx = np.argsort(-np.abs(corr))
        gene_list = fadata_sub.var_names[rank_idx].tolist()

        edge_dict[edge_key] = gene_list[:top_n]

    return edge_dict

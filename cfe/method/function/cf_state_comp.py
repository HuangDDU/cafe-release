import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler, normalize


def cf_state_comp(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # 1. prepare data
    adata = adata.copy()
    cell_ids = adata.obs.index

    # 2. preprocess
    ndim = parameters["ndim"]
    sc.pp.pca(adata, n_comps=ndim)

    # 3. execute method
    # extract pca results as state transition probabilities
    X_pca = adata.obsm["X_pca"]
    X_pca_scaled = MinMaxScaler().fit_transform(X_pca)  # Normalization
    pseudotime = X_pca_scaled[:, parameters["component"]-1]  # specified component for pseudotime
    comp_column_list = [f"comp_{i}" for i in range(1, ndim+1)]  # the first ndim components correspond to n states
    # The normalized PCA result is used as the state transition probability, range of [0,1]
    end_state_probabilities = pd.DataFrame(
        columns=comp_column_list,
        data=normalize(X_pca_scaled, norm="l1"),  # 归一化后的PCA结果作为状态转移概率, 从from的概率为0
        index=cell_ids,
    )
    end_state_probabilities["cell_id"] = cell_ids
    end_state_probabilities = end_state_probabilities[["cell_id"] + comp_column_list]

    # 4. save results
    wrapper_type = parameters.get("wrapper_type", "probability")
    if wrapper_type == "lineage":
        # for lineage wrapper
        cluster_key = parameters.get("cluster_key", "clusters")
        trajectory_dict = {
            "probability": end_state_probabilities[end_state_probabilities.columns[1:]],
            "cluster_key": cluster_key,  # TODO: 暂时指定obs[cluster_key]标签下的终端状态
            "wrapper_type": "lineage"
        }
    else:
        # for probability wrapper
        trajectory_dict = trajectory_dict = {
            "end_state_probabilities": end_state_probabilities,
            "pseudotime": pseudotime,
        }

    return trajectory_dict

import anndata as ad
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler, normalize


def state_comp(
    adata: ad.AnnData,
    repreprocess: bool = True,
    n_comps: int = 2,
    basis: str = "X_pca",
    pseudotime_index: int = 1,
    wrapper_type: str = "probability",
    cluster_key: str = "clusters",
):
    # 1. preprocess
    if repreprocess:
        sc.pp.pca(adata, n_comps=n_comps)
    cell_ids = adata.obs.index

    # 2. execute method
    # extract embedding results as state transition probabilities
    X_emb = adata.obsm[basis][:, :n_comps]
    X_emb_scaled = MinMaxScaler().fit_transform(X_emb)  # Normalization
    pseudotime = X_emb_scaled[:, pseudotime_index]  # specified component for pseudotime
    comp_column_list = [f"comp_{i}" for i in range(1, n_comps + 1)]  # the first ndim components correspond to n states
    # The normalized PCA result is used as the state transition probability, range of [0,1]
    end_state_probabilities = pd.DataFrame(
        columns=comp_column_list,
        data=normalize(X_emb_scaled, norm="l1"),  # 归一化后的PCA结果作为状态转移概率, 从from的概率为0
        index=cell_ids,
    )
    end_state_probabilities["cell_id"] = cell_ids
    end_state_probabilities = end_state_probabilities[["cell_id"] + comp_column_list]

    # 3,4. extract and save results
    if wrapper_type == "lineage":
        # for lineage wrapper
        trajectory_dict = {
            "probability": end_state_probabilities[end_state_probabilities.columns[1:]],
            "cluster_key": cluster_key,  # TODO: 暂时指定obs[cluster_key]标签下的终端状态
            "wrapper_type": "lineage",
        }
    else:
        # for probability wrapper
        trajectory_dict = trajectory_dict = {
            "end_state_probabilities": end_state_probabilities,
            "pseudotime": pseudotime,
        }

    return trajectory_dict


def cf_state_comp(
    adata: ad.AnnData,
    prior_information: dict = None,
    parameters: dict = None,
    **kwargs,
):
    if (prior_information is None) and (parameters is None):
        # for new backend call, function(**kwargs)
        return state_comp(adata, **kwargs)
    else:
        # for old backend call, function(prior_information, parameters)
        parameters.update(prior_information)
        return state_comp(adata, **parameters)

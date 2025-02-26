import pandas as pd
import anndata as ad
import scanpy as sc


def cf_state_comp(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # 1. 数据构造
    adata = adata.copy()
    cell_ids = adata.obs.index

    # 2. 执行PCA
    ndim = parameters["ndim"]
    sc.pp.pca(adata, n_comps=ndim)

    # 3. 提取PCA结果作为状态转移概率
    pseudotime = adata.obsm["X_pca"][:, parameters["component"]-1]  # 伪时间用指定的分量
    comp_column_list = [f"comp_{i}" for i in range(1, ndim+1)]  # 前ndim个分量对应n个状态
    end_state_probabilities = pd.DataFrame(
        columns=comp_column_list,
        data=adata.obsm["X_pca"],
        index=cell_ids,
    )
    end_state_probabilities["cell_id"] = cell_ids
    end_state_probabilities = end_state_probabilities[["cell_id"] + comp_column_list]

    # 4. 结果封装保存
    trajectory_dict = trajectory_dict = {
        "end_state_probabilities": end_state_probabilities,
        "pseudotime": pseudotime,
    }

    return trajectory_dict

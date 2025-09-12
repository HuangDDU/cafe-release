import anndata as ad
import cellrank as cr

# import numpy as np
import pandas as pd
import scanpy as sc


# TODO: update parameters
def cf_cellrank(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # 1. prepare data
    copy = parameters.get("copy", True)
    adata = adata.copy() if copy else adata
    cell_ids = adata.obs.index

    # 2. preprocess and execute method simutaneously with pca
    repreprocess = parameters.get("repreprocess", True)
    # n_comps = parameters["ndim"]
    # knn = knn = parameters["knn"]
    if repreprocess:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)

    # 3. extract results
    kernel = parameters.get("kernel", "connectivity")
    n_states = parameters.get("n_states", 10)
    n_terminal_states = parameters.get("n_terminal_states", 4)
    initial_states = parameters.get("initial_states", None)
    terminal_states = parameters.get("terminal_states", None)
    cluster_key = parameters.get("cluster_key", "clusters")
    if kernel == "connectivity":
        # kernel
        ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
        ck.compute_transition_matrix()
        # estimator
        g = cr.estimators.GPCCA(ck)
        g.fit(n_states=n_states, cluster_key=cluster_key)
        if initial_states is not None:
            g.set_initial_states(states=initial_states)
        if terminal_states is not None:
            # set terminal states mannually
            g.set_terminal_states(states=terminal_states)
        else:
            g.predict_terminal_states(method="top_n", n_states=n_terminal_states)
        g.compute_fate_probabilities()
        # lineage object
        lineage = g._fate_probabilities
        end_state_probabilities = pd.DataFrame(lineage.__array__(), columns=lineage.names)

        # macrostate_df = pd.DataFrame(
        #     g.macrostates_memberships.__array__(), columns=g.macrostates.cat.categories.tolist()
        # )
        # macrostate_list = macrostate_df.idxmax(axis=1).tolist()
    else:
        # TODO: Other kernel in parameters
        pass

    # 4. save results
    wrapper_type = parameters.get("wrapper_type", "probability")
    if wrapper_type == "lineage":
        # for lineage wrapper
        # 直接使用新的宏状态标签
        # trajectory_dict = {
        #     "probability": end_state_probabilities,
        #     "cluster_key": None,  # TODO: 暂时指定obs[cluster_key]标签下的终端状态
        #     "new_cluster_list": macrostate_list,
        #     "wrapper_type": "lineage"
        # }
        # 去除'_数字'后缀

        # TODO: 合并去除_后缀后，取平均相同的列
        end_state_probabilities.columns = end_state_probabilities.columns.str.replace(r"_\d+", "")
        trajectory_dict = {
            "probability": end_state_probabilities,
            "cluster_key": cluster_key,
            "new_cluster_list": None,
            "wrapper_type": "lineage",
        }
    else:
        # for probability wrapper
        end_state_probabilities["cell_id"] = cell_ids
        trajectory_dict = {
            "end_state_probabilities": end_state_probabilities,
        }

    return trajectory_dict

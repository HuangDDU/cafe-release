import re

import anndata as ad
import cellrank as cr

# import numpy as np
import pandas as pd

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cafe environment
    from cafe.method.function.method_decorator import method_info
    from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="cellrank",
    version="0.0.1",
    description="CellRank 2: unified fate mapping in multiview single-cell data",
    wrapper_type="velocity",
    doi="10.1038/s41592-024-02303-9",
    github_url="https://github.com/theislab/cellrank",
    use_gpu=False,
    cpu_parallelization=True,
)
def cellrank(
    adata: ad.AnnData,
    cluster: str,
    repreprocess: bool = True,
    wrapper_type: str = "probability",
    kernel: str = "connectivity",
    kernel_params: dict = {},
    initial_states=None,
    terminal_states=None,
    fit_kwargs: dict = {},
    predict_terminal_states_kwargs: dict = {},
    using_macrostate: bool = True,
):
    """CellRank 2: unified fate mapping in multiview single-cell data

    Args:
        adata (ad.AnnData): AnnData object.
        repreprocess (bool, optional): Whether to repreprocess the anndata object.

    Returns:
        dict: trajectory dict with keys about velocity
    """
    # 1.  preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="scanpy")

    # 2. execute method
    # kernel
    if kernel == "connectivity":
        kernel_obj = cr.kernels.ConnectivityKernel(adata, **kernel_params)
    elif kernel == "velocity":
        if "velocity" not in adata.layers:
            # TODO: check and calculate velocity adata.layer["velocity"] first
            raise ValueError("adata.layers['velocity'] not found, please calculate velocity first.")
        kernel_obj = cr.kernels.VelocityKernel(adata, **kernel_params).compute_transition_matrix()
    else:
        # TODO: Other kernel in parameters
        kernel_obj = None
    # TODO: complex kernel with multiple views
    kernel_obj.compute_transition_matrix()

    # estimator
    g = cr.estimators.GPCCA(kernel_obj)
    # identify macrostates, related parameters are in fit_kwargs
    if fit_kwargs.get("cluster_key") is None:
        fit_kwargs["cluster_key"] = cluster
        print(f"use default cluster key: {cluster}")
    if fit_kwargs.get("n_states") is None:
        cluster = fit_kwargs["cluster_key"]
        n_states = len(adata.obs[cluster].cat.categories)
        fit_kwargs["n_states"] = n_states
        print(f"set n_states={n_states} according to cluster key({cluster})")
    g.fit(**fit_kwargs)
    macrostates = g.macrostates.cat.categories.tolist()  # valid macro states
    # set initial and terminal states
    if initial_states is not None:
        initial_states = [macrostate for macrostate in macrostates if re.sub(r"_\d+$", "", macrostate) in initial_states]
        g.set_initial_states(states=initial_states)
        print(f"set initial states({initial_states}) mannually")
    if terminal_states is not None:
        # set terminal states mannually
        terminal_states = [macrostate for macrostate in macrostates if re.sub(r"_\d+$", "", macrostate) in terminal_states]
        g.set_terminal_states(states=terminal_states)
        print(f"set terminal states({terminal_states}) mannually")
    else:
        g.predict_terminal_states(**predict_terminal_states_kwargs)
        print("predict terminal states automatically")
    # compute fate probabilities
    g.compute_fate_probabilities()
    # extract lineage object
    lineage = g._fate_probabilities
    end_state_probabilities = pd.DataFrame(lineage.__array__(), columns=lineage.names, index=adata.obs.index)
    adata.obsm["lineages_fwd"] = lineage
    adata.obs[end_state_probabilities.columns] = end_state_probabilities
    # macrostate
    macrostate_df = pd.DataFrame(g.macrostates_memberships.__array__(), columns=g.macrostates.cat.categories.tolist())
    macrostate_list = macrostate_df.idxmax(axis=1).tolist()

    # 3. extract results
    trajectory_dict = {"wrapper_type": wrapper_type}
    if wrapper_type == "lineage":
        if using_macrostate:
            # macrostate as cluster with suffix number
            macrostate_df = pd.DataFrame(g.macrostates_memberships.__array__(), columns=g.macrostates.cat.categories.tolist())
            macrostate_list = macrostate_df.idxmax(axis=1).tolist()
            trajectory_dict["new_cluster_list"] = macrostate_list
        else:
            # raw cluster, remove suffix number
            end_state_probabilities.columns = end_state_probabilities.columns.str.replace(r"_\d+", "")  # remove suffix number
            trajectory_dict["cluster_key"] = cluster
        trajectory_dict["probability"] = end_state_probabilities

    else:
        # for probability wrapper
        trajectory_dict["end_state_probabilities"] = end_state_probabilities

    # 4. save results
    return trajectory_dict


# def extract_trajectory_dict(adata, wrapper_type="lineage"):

#     trajectory_dict = {
#         "wrapper_type": "velocity",
#         "velocity": None,
#         "velocity_graph": None,
#         "velocity_graph_neg": None,
#         "velocity_embedding": adata.obsm[f"velocity_{basis[2:]}"],
#         "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
#         "obs_index": adata.obs.index,
#         "var_index": adata.var.index,
#     }
#     return trajectory_dict

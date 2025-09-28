import anndata as ad
import dynamo as dyn
import numpy as np

# import scipy.sparse as sp
import scvelo as scv

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cfe environment
    from cfe.method.function.method_decorator import method_info
    from cfe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="dynamo",
    version="0.0.1",
    description="Dynamo: Mapping Transcriptomic Vector Fields of Single Cells",
    wrapper_type="velocity",
    doi="10.1016/j.cell.2021.12.045",
    github_url="https://github.com/aristoteleo/dynamo-release",
)
def dynamo(
    adata: ad.AnnData,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = {},
    dynamics_kwargs: dict = {},
    cell_velocities_kwargs: dict = {},
):
    """Dynamo: Mapping Transcriptomic Vector Fields of Single Cells"""
    # 1. preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="dynamo", **repreprocess_kwargs)

    # 2. execute method
    dyn.tl.dynamics(adata, **dynamics_kwargs)
    dyn.tl.cell_velocities(adata, **cell_velocities_kwargs)
    adata.layers["velocity"] = adata.layers["velocity_S"].toarray()  # csr sparse matrix to array
    if np.min((adata.obsp["distances"] > 0).sum(1).A1) == 0:
        print("neighbor graph seems to be corrupted, recompute neighbor graph.")
        scv.pp.neighbors(adata)  # recompute neighbor graph if some points have no neighbors
    scv.tl.velocity_graph(adata)
    # TODO: compute velocity graph later in wrapper function: 'add_trajectory_velocity'

    # TODO: use raw dynamo velocity graph, but 'velocity_graph_neg' can't set accurately for paga.
    # velocity_graph = adata.obsp["pearson_transition_matrix"]
    # adata.uns["velocity_graph"] = velocity_graph
    # # adata.uns["velocity_graph_neg"] = sp.csr_matrix(velocity_graph.shape, dtype=velocity_graph.dtype) # fill 0
    # adata.uns["velocity_graph_neg"] = sp.csr_matrix(np.ones(velocity_graph.shape)/velocity_graph.shape[1]) # fill avg value

    # 3,4. extract and save results
    trajectory_dict = {
        "wrapper_type": "velocity",
        "velocity": adata.layers["velocity"],
        "velocity_graph": adata.uns["velocity_graph"],
        "velocity_graph_neg": adata.uns["velocity_graph_neg"],
        "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,
    }

    return trajectory_dict

import anndata as ad
import dynamo as dyn

# import numpy as np
# import scipy.sparse as sp
# import scanpy as sc
# import scvelo as scv

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cafe environment
    from cafe.method.function.method_decorator import method_info
    from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="dynamo",
    version="0.0.1",
    description="Dynamo: Mapping Transcriptomic Vector Fields of Single Cells",
    wrapper_type="velocity",
    doi="10.1016/j.cell.2021.12.045",
    github_url="https://github.com/aristoteleo/dynamo-release",
    use_gpu=False,
    cpu_parallelization=True,
)
def dynamo(
    adata: ad.AnnData,
    basis: str,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = {},
    moment: bool = True,
    n_neighbors: int = 30,
    dynamics_kwargs: dict = {},
    cell_velocities_kwargs: dict = {},
):
    """Dynamo: Mapping Transcriptomic Vector Fields of Single Cells

    Args:
        adata (ad.AnnData): AnnData object.
        repreprocess (bool, optional): Whether to repreprocess the anndata object.
        repreprocess_kwargs (dict, optional): Parameter dict for repreprocess pipeline with dynamo style.
        dynamics_kwargs (dict, optional): Parameter dict for cell dynamics high dimensional velocity calculation, refer to [dyn.tl.dynamics](https://dynamo-release.readthedocs.io/en/latest/api/reference/dynamo.tl.dynamics.html#dynamo.tl.dynamics).
        cell_velocities_kwargs (dict, optional): Parameter dict for cell low dimensional velocity calculation, refer to [dynamo.tl.cell_velocities](https://dynamo-release.readthedocs.io/en/latest/api/reference/dynamo.tl.cell_velocities.html#dynamo.tl.cell_velocities).

    Returns:
        dict: trajectory dict with keys about velocity

    """
    # 1. preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="dynamo", **repreprocess_kwargs)
        dyn.tl.moments(adata)  # make sure moments is computed
        adata.layers["Ms"] = adata.layers["M_s"]  # extract moment matrix for transition matrix calculation
        adata.layers["Mu"] = adata.layers["M_u"]
        dyn.tl.neighbors(adata, n_neighbors=n_neighbors)  # recompute neighbors
    basis = basis[2:]  # remove "X_"
    # 2. execute method
    # dynamo core function
    dyn.tl.dynamics(adata, **dynamics_kwargs)
    dyn.tl.cell_velocities(adata, basis=basis, **cell_velocities_kwargs)  # scv.tl.velocity_graph(adata)
    # velocity_key = "velocity"
    # adata.layers[velocity_key] = adata.layers["velocity_S"].toarray()  # extract velocity matrix
    # adata.var[f"{velocity_key}_genes"] = adata.var["use_for_transition"]  # extract velocity gene
    adata = adata[:, adata.var["use_for_transition"]]  # extract velocity gene
    velocity_embedding = adata.obsm[f"velocity_{basis}"]

    # 3,4. extract and save results
    trajectory_dict = {
        "wrapper_type": "velocity",
        "velocity": None,
        "velocity_graph": None,
        "velocity_graph_neg": None,
        "velocity_embedding": velocity_embedding,
        "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,
    }

    return trajectory_dict

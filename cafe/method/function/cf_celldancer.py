import anndata as ad

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cafe environment
    from cafe.method.function.method_decorator import method_info
    from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="celldancer",
    version="0.0.1",
    description="CellDancer: Estimating Cell-dependent RNA Velocity",
    wrapper_type="velocity",
    doi="10.1038/s41587-023-01728-5",
    github_url="https://github.com/GuangyuWangLab2021/cellDancer",
    use_gpu=True,
    cpu_parallelization=True,
    available=True,
)
def celldancer(
    adata: ad.AnnData,
    cluster: str,
    basis: str,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = {},
    velocity_kwargs: dict = {},
    compute_cell_velocity_kwargs: dict = {},
):
    """CellDancer: Estimating Cell-dependent RNA Velocity

    Args:
        adata (ad.AnnData): AnnData object
        repreprocess (bool, optional): Whether to repreprocess the anndata object.
        repreprocess_kwargs (dict, optional): Parameter dict for repreprocess pipeline.

    Returns:
        dict: trajectory dict with keys about velocity, only velocity_embedding is available
    """

    import celldancer as cd

    # 1. preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="scvelo", **repreprocess_kwargs)

    # 2. execute method
    # transfer adata to cellDancer format
    cellDancer_df = cd.utilities.adata_to_df_with_embed(
        adata,
        us_para=["Mu", "Ms"],
        cell_type_para=cluster,
        embed_para=basis,
    )
    loss_df, cellDancer_df = cd.velocity(cellDancer_df, **velocity_kwargs)
    cellDancer_df = cd.compute_cell_velocity(cellDancer_df=cellDancer_df, **compute_cell_velocity_kwargs)
    # loss_df, cellDancer_df = cd.velocity(cellDancer_df)
    # cellDancer_df = cd.compute_cell_velocity(cellDancer_df=cellDancer_df, projection_neighbor_size=100) # compute by transciption parameter

    # 3. extract results
    velocity_df = cellDancer_df.groupby("cellID").first()[["velocity1", "velocity2"]].fillna(0)
    adata.obsm[f"velocity_{basis[2:]}"] = velocity_df.loc[adata.obs.index].values  # align index
    # celldancer generate many zero velocity cells, only extracted valid velocity cell to construct trajectory.
    adata = adata[~((adata.obsm[f"velocity_{basis[2:]}"] == 0).all(axis=1))].copy()
    trajectory_dict = extract_trajectory_dict(adata, basis=basis)

    # 4. save results
    return trajectory_dict


def extract_trajectory_dict(adata, basis="X_umap"):
    trajectory_dict = {
        "wrapper_type": "velocity",
        "velocity": None,
        "velocity_graph": None,
        "velocity_graph_neg": None,
        "velocity_embedding": adata.obsm[f"velocity_{basis[2:]}"],
        "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,
    }
    return trajectory_dict

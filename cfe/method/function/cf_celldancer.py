import anndata as ad

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cfe environment
    from cfe.method.function.method_decorator import method_info
    from cfe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="celldancer",
    version="0.0.1",
    description="CellDancer: Estimating Cell-dependent RNA Velocity",
    wrapper_type="velocity",
    doi="10.1038/s41587-023-01728-5",
    github_url="https://github.com/GuangyuWangLab2021/cellDancer",
)
def celldancer(
    adata: ad.AnnData,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = {},
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
        cell_type_para="clusters",
        embed_para="X_umap",
    )
    loss_df, cellDancer_df = cd.velocity(cellDancer_df, permutation_ratio=0.5, n_jobs=24)
    cellDancer_df = cd.compute_cell_velocity(cellDancer_df=cellDancer_df, projection_neighbor_size=100)

    # 3. extract results
    velocity_embedding = cellDancer_df.groupby("cellID").first()[["velocity1", "velocity2"]].fillna(0).values

    # 4. save results
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

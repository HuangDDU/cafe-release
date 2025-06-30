import anndata as ad
import scanpy as sc


def sample(
    adata: ad.AnnData,
    n_obs: int = -1,
    save_cell_list: list = None,
):
    # 采样细胞，并保留指定细胞
    if n_obs > 0:
        if len(save_cell_list) > 0:
            adata_save = adata[save_cell_list, :].copy()
            n_obs -= len(adata_save)
            adata = adata[~(adata.obs.index.map(lambda x: x in save_cell_list))].copy()
        else:
            adata_save = None

        sc.pp.subsample(adata, n_obs=n_obs)  # subsample

    if adata_save is not None:
        adata = sc.concat([adata_save, adata])

    # TODO: 以后对于milestone_network结构的采样，实现真正对于FateAnndata的采样
    return adata

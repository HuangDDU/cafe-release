import anndata as ad
import scanpy as sc

from .._logging import logger


def subsample(
    adata: ad.AnnData,
    n_obs: int = -1,
    save_cell_list: list = [],
    save_cluster_key: str = None,
):
    """subsample from adata, save sa

    Args:
        adata (ad.AnnData): raw AnnData
        n_obs (int, optional): _description_. Defaults to -1.
        save_cell_list (list, optional): key cells such as start and end point.
        save_cluster_key (str, optional): _description_. Defaults to None.

    Returns:
        adata: subsample AnnData
    """

    if n_obs > 0:
        adata = adata.copy()
        adata_list = []
        if len(save_cell_list) > 0:
            # sample mannually specific cells
            logger.debug("sample mannually specific cells")
            adata_save = adata[save_cell_list, :].copy()
            adata_list.append(adata_save)
            n_obs -= len(save_cell_list)
            adata = adata[~(adata.obs.index.map(lambda x: x in save_cell_list))].copy()
        if save_cluster_key is not None:
            # sample 1 cells for every cluster
            # TODO: should exclude above mannual specific cells
            logger.debug("sample 1 cells for every cluster")
            cluster_cell_list = adata.obs.groupby(save_cluster_key).head(1).index.tolist()
            adata_save2 = adata[cluster_cell_list, :].copy()
            adata_list.append(adata_save2)
            n_obs -= len(cluster_cell_list)
            adata = adata[~(adata.obs.index.map(lambda x: x in cluster_cell_list))].copy()
        if n_obs > 0:
            # sample random cells
            logger.debug("sample random cells")
            sc.pp.subsample(adata, n_obs=n_obs)  # subsample
            adata_list.append(adata)
        elif n_obs == 0:
            logger.debug("skip sampling random cells ")
        else:
            raise Exception("n_obs is too small")

        adata_subsample = sc.concat(adata_list)

        # TODO: the subsample only can be used before trajectory addition, need subsample milestone and waypoint for flexible usage
        subsample_milestone()
        subsample_waypoint()

        return adata_subsample
    else:
        return adata


def subsample_milestone():
    # TODO: subsample cells from milestone: percentage, progression
    pass


def subsample_waypoint():
    # TODO: subsample cells from waypoint: ???
    pass

import anndata as ad
import scanpy as sc


def comp1(
    adata: ad.AnnData,
    repreprocess: bool = True,
    pca_ndim: int = 5,
    basis: str = "X_pca",
    component: int = 1,
    **kwargs,
):
    """_summary_

    Args:
        adata (ad.AnnData): _description_
        repreprocess (bool, optional): _description_. Defaults to True.
        pca_ndim (int, optional): _description_. Defaults to 5.
        basis (str, optional): _description_. Defaults to "X_pca".
        component (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # 1,2  preprocess and execute method simutaneously with pca
    if repreprocess and (basis == "X_pca"):
        sc.pp.pca(adata, n_comps=pca_ndim)

    # 3. extract results
    pseudotime = adata.obsm[basis][:, component - 1]

    # 4. save results
    trajectory_dict = {
        "wrapper_type": "linear",
        "pseudotime": pseudotime,
    }

    return trajectory_dict


def cf_comp1(
    adata: ad.AnnData,
    prior_information: dict = None,
    parameters: dict = None,
    **kwargs,
):
    if (prior_information is None) and (parameters is None):
        # for new backend call, function(**kwargs)
        return comp1(adata, **kwargs)
    else:
        # for old backend call, function(prior_information, parameters)
        parameters.update(prior_information)
        return comp1(adata, **parameters)

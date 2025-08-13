import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv


def paga(
    adata: ad.AnnData,
    start_id: str,
    repreprocess: bool = True,
    filter_and_normalize_kwargs: dict = {},
    neighbors_kwargs: dict = {},
    cluster_key: str = "clusters",
    n_dcs: int = 15,
    connectivity_cutoff=0.5,
    **kwargs,
):
    """PAGA trajectory inference method.

    Args:
        adata (ad.AnnData): AnnData object
        start_id (str): Starting cell ID for pseudotime calculation.
        repreprocess (bool, optional): whether reprocess the anndata object, including feature selection, normalization, scale, pca and neighbor computation. Defaults to True.
        filter_and_normalize_kwargs (dict, optional): Parameters for preprocess in scvelo style, refer to "scvelo.pp.filter_and_normalize"(https://scvelo.readthedocs.io/en/stable/scvelo.pp.filter_and_normalize.html). Defaults to {}.
        neighbors_kwargs (dict, optional):  Parameters for neighbor construction in scanpy style, refer to "scanpy.pp.neighbors"(https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pp.neighbors.html). Defaults to {}.
        cluster_key (str, optional): Cluster column name in adata.obs. Defaults to "clusters".
        n_dcs (int, optional): Number of diffusion components. Defaults to 15.
        connectivity_cutoff (float, optional): Cutoff for the connectivity matrix. Defaults to 0.5.

    Returns:
        dict: Trajectory results including branch network, branches, and progressions.
    """
    # 1. preprocess
    if repreprocess:
        scv.pp.filter_and_normalize(adata, **filter_and_normalize_kwargs)
        sc.pp.neighbors(adata, **neighbors_kwargs)
    sc.tl.diffmap(adata)

    # 2. execute method
    sc.tl.paga(adata, groups=cluster_key)
    # set start porint for dpt
    adata.uns["iroot"] = np.where(adata.obs.index == start_id)[0][0]
    sc.tl.dpt(adata, n_dcs=n_dcs)

    # 3. extract results
    # (1) parameters for results extracting
    epsilon = 1e-3  # a very small scaling values
    branch_ids = adata.obs[cluster_key].unique().to_list()
    # (2) branches
    branches = pd.DataFrame(
        {
            "branch_id": branch_ids,
            "directed": True,
        }
    )
    branches["length"] = (
        adata.obs[[cluster_key, "dpt_pseudotime"]]
        .groupby(cluster_key)
        .apply(lambda x: x["dpt_pseudotime"].max() - x["dpt_pseudotime"].min() + epsilon)
        .reset_index()[0]
    )
    # (3) branch_network
    branch_network = (
        pd.DataFrame(
            np.triu(adata.uns["paga"]["connectivities"].todense(), k=0),  # keep the upper triangular matrix
            index=adata.obs[cluster_key].cat.categories,
            columns=adata.obs[cluster_key].cat.categories,
        )
        .stack()
        .reset_index()
    )
    branch_network.columns = ["from", "to", "length"]
    branch_network = branch_network[branch_network["length"] >= connectivity_cutoff]  # set threshold to filter insignificant edges
    average_pseudotime_dict = adata.obs.groupby(cluster_key)["dpt_pseudotime"].mean()

    def modify_milestone_network_direction(x):
        if average_pseudotime_dict[x["from"]] <= average_pseudotime_dict[x["to"]]:
            return x
        else:
            x["from"], x["to"] = x["to"], x["from"]
            return x

    branch_network.apply(modify_milestone_network_direction, axis=1)  # Adjust the direction of the edge
    # sort edges by "from" and "to" columns to facilitate subsequent milestone numbering
    branch_network["from_pseudotime"] = branch_network["from"].apply(lambda x: average_pseudotime_dict[x])
    branch_network["to_pseudotime"] = branch_network["to"].apply(lambda x: average_pseudotime_dict[x])
    branch_network = branch_network.sort_values(["from_pseudotime", "to_pseudotime"])
    branch_network = branch_network[["from", "to"]].reset_index(drop=True)
    # (4) branch_progressions
    branch_progressions = pd.DataFrame({"cell_id": adata.obs.index, "branch_id": adata.obs[cluster_key], "percentage": adata.obs["dpt_pseudotime"]})
    # sort cells by pseudo time within the branch
    branch_progressions["percentage"] = (
        branch_progressions.groupby("branch_id")["percentage"].apply(lambda x: (x - x.min()) / (x.max() - x.min() + epsilon)).values
    )
    branch_progressions

    # 4. save results
    trajectory_dict = {
        "branch_network": branch_network,
        "branches": branches,
        "branch_progressions": branch_progressions,
    }
    return trajectory_dict


def cf_paga(
    adata: ad.AnnData,
    prior_information: dict = None,
    parameters: dict = None,
    **kwargs,
):
    if (prior_information is None) and (parameters is None):
        # for new backend call, function(**kwargs)
        return paga(adata, **kwargs)
    else:
        # for old backend call, function(prior_information, parameters)
        parameters.update(prior_information)
        return paga(adata, **parameters)

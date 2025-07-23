import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv


def cf_paga(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # 1. extract prior information and parameters
    start_id = prior_information["start_id"]
    repreprocess = parameters["repreprocess"]
    filter_and_normalize_kwargs = parameters["filter_and_normalize_kwargs"]
    neighbors_kwargs = parameters["neighbors_kwargs"]
    cluster_key = parameters["cluster_key"]
    n_dcs = parameters["n_dcs"]
    connectivity_cutoff = parameters["connectivity_cutoff"]

    # 2. preprocess
    if repreprocess:
        scv.pp.filter_and_normalize(adata, **filter_and_normalize_kwargs)
        sc.pp.neighbors(adata, **neighbors_kwargs)
    sc.tl.diffmap(adata)

    # 3. execute method
    sc.tl.paga(adata, groups=cluster_key)
    # set start porint for dpt
    adata.uns["iroot"] = np.where(adata.obs.index == start_id)[0][0]
    sc.tl.dpt(adata, n_dcs=n_dcs)

    # 4. extract results
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

    # 5. save results
    trajectory_dict = {
        "branch_network": branch_network,
        "branches": branches,
        "branch_progressions": branch_progressions,
    }
    return trajectory_dict

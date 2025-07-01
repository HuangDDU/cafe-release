import anndata as ad
import numpy as np
import pandas as pd
import scvelo as scv


def cf_scvelo(adata: ad.AnnData, prior_information: dict = {}, parameters: dict = {}):
    # ref: https://scvelo.readthedocs.io/en/stable/VelocityBasics.html
    cluster_key = prior_information.get("cluster_key", "clusters")

    # 1. prepare data
    adata = adata.copy()

    # 2. preprocess
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # 3. execute method
    scv.tl.velocity(adata)  # compute high dimensional velocity
    scv.tl.velocity_graph(adata)  # compute transition probability
    # scv.pl.velocity_embedding_stream(adata, basis="umap", show=False)  # NOTE: compared with plot_wrapper

    # 4. PAGA calculation of milestone network directed graph
    milestone_id_list = list(adata.obs[cluster_key].cat.categories)

    adata.uns["neighbors"]["distances"] = adata.obsp["distances"]
    adata.uns["neighbors"]["connectivities"] = adata.obsp["connectivities"]
    scv.tl.paga(adata, groups=cluster_key)
    df = scv.get_df(adata, "paga/transitions_confidence", precision=2).T
    df.index = milestone_id_list
    df.columns = milestone_id_list

    milestone_network = (
        df.reset_index().rename(columns={"index": "from"}).melt(id_vars="from", var_name="to", value_name="length").query("`length` > 0")
    )
    milestone_network["length"] = 1  # Temporarily set uniformly to 1
    milestone_network["directed"] = True

    # 4. extract results
    obs = adata.obs.reset_index()  # change index
    X_emb = adata.obsm["X_umap"]
    milestone_emb = np.array(list(obs.groupby(cluster_key).apply(lambda x: X_emb[list(x.index)].mean(axis=0))))
    milestone_emb = pd.DataFrame(milestone_emb, index=milestone_id_list)
    # TODO: minimal anndata for plot_wrapper
    velocity_adata = adata

    # 5. save results
    trajectory_dict = {
        # for trajectory transform
        "milestone_network": milestone_network,
        "X_emb": X_emb,
        "milestone_emb": milestone_emb,
        # for wrapper plot
        # "velocity": adata.layers["velocity"],  # high dimensional velocity matrix(n_obs*n_obs)
        # "neighbors": adata.uns["neighbors"],
        # "cell_index": adata.obs.index,
        # "gene_index": adata.var.index,
        "velocity_adata": velocity_adata,
    }

    # TODO: need update wrapper
    return trajectory_dict

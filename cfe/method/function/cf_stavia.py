import anndata as ad
import pandas as pd

try:
    # for docker
    from method_decorator import method_info

    # from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cfe environment
    from cfe.method.function.method_decorator import method_info

    # from cfe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="stavia",
    version="0.0.1",
    description="StaVia: spatially and temporally aware cartography with higher-order random walks for cell atlases",
    wrapper_type="cluster",
    doi="10.1186/s13059-024-03347-y",
    github_url="https://github.com/ShobiStassen/VIA",
)
def stavia(
    adata: ad.AnnData,
    cluster: str,
    start_cell: str,
    repreprocess: bool = True,
    data_basis: str = "X_pca",
    ncomps: int = 30,
    via_kwargs: dict = {},
    prune_milestone: bool = True,  # whether to prune the milestone network
):
    """StaVia: spatially and temporally aware cartography with higher-order random walks for cell atlases
    Returns:
        dict: trajectory dict with keys about cluster wrapper
    """
    # ref: https://pyvia.readthedocs.io/en/latest/notebooks/ViaJupyter_scRNA_Hematopoiesis.html
    import igraph as ig
    import pyVIA.core as via

    # 1. preprocess
    # 2. execute method
    via_object = via.VIA(
        data=adata.obsm[data_basis][:, :ncomps],
        true_label=adata.obs[cluster],
        root_user=start_cell,
        **via_kwargs,
    )
    via_object.run_VIA()

    # 3. extract results
    if prune_milestone:
        # ref: plot_trajectory_curves (https://github.com/ShobiStassen/VIA/blob/master/VIA/plotting_via.py#L3100)
        super_edgelist = via_object.edgelist_maxout
        super_cluster_labels = via_object.labels  # 细胞归属里程碑标签
        final_super_terminal = via_object.terminal_clusters
        super_root = via_object.root[0]
        G_orange = ig.Graph(n=len(set(super_cluster_labels)), edges=super_edgelist)
        # 保存从根节点到各终端状态的最短路径的边
        ll_ = []
        for fst_i in final_super_terminal:
            path_orange = G_orange.get_shortest_paths(super_root, to=fst_i)[0]
            len_path_orange = len(path_orange)
            for enum_edge, edge_fst in enumerate(path_orange):
                if enum_edge < (len_path_orange - 1):
                    ll_.append((edge_fst, path_orange[enum_edge + 1]))
        edgelist = list(set(ll_))
    else:
        edgelist = via_object.edgelist_maxout
    edgelist = [[str(item[0]), str(item[1])] for item in edgelist]
    milestone_network = pd.DataFrame(
        data=edgelist,
        columns=["from", "to"],
    )
    milestone_network["length"] = 1
    milestone_network["directed"] = True
    cluster_milestones = [str(i) for i in via_object.labels]

    # 4. save results
    trajectory_dict = {
        "wrapper_type": "cluster",
        "milestone_network": milestone_network,
        "cluster": cluster_milestones,
    }
    return trajectory_dict

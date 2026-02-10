import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, pairwise_distances, silhouette_score

from .._logging import logger
from ..data import FateAnnData


# TODO: Escort metrics: https://github.com/xiaorudong/Escort
# ============================ inspired by StaVia ============================
def get_embedding_graph(
    fadata: FateAnnData,
    basis: str,
    graph_type: str = "geodesic",
    recompute_neighbors: bool = True,
    directed: bool = False,
    n_neighbors: int = 5,
    normalized: bool = True,
    return_type: str = "graph",
):
    # get graph in embedding space
    if basis is None:
        basis = fadata.prior_information.get("basis")

    embedding_graph_key = f"{basis}_{graph_type}_graph"
    if recompute_neighbors or (embedding_graph_key not in fadata.embedding_cache):
        if graph_type == "euclidean":
            logger.debug("build euclidean graph...")
            emb = fadata.obsm[basis]
            distances_matrix = pairwise_distances(emb, emb)
        else:
            logger.debug("build geodesic graph...")
            # compute neighbors on embedding
            fadata_new = ad.AnnData(fadata.obsm[basis])
            if n_neighbors is None:
                n_neighbors = fadata.uns["neighbors"]["params"]["n_neighbors"][0]
                logger.debug(f"n_neighbors is None, use default n_neighbors:{n_neighbors} from fadata.uns")

            sc.pp.neighbors(fadata_new, n_neighbors=n_neighbors)
            # build graph
            if directed:
                distances_matrix = fadata_new.obsp["distances"]
            else:
                connectivities = fadata_new.obsp["connectivities"]
                logger.debug(f"connectivities : {connectivities}")
                rows, cols = connectivities.nonzero()
                connectivities[rows, cols] = 1 / connectivities[rows, cols]
                distances_matrix = connectivities
        if normalized:
            # normalize distances to [0,1] to ignore different embedding scale
            max_distance = distances_matrix.max()
            min_distance = distances_matrix.min()
            distances_matrix = (distances_matrix - min_distance) / (max_distance - min_distance)
        G = nx.Graph(distances_matrix, create_using=nx.DiGraph if directed else nx.Graph)
        # set graph into cache
        embedding_basis_cache = {
            "distances_matrix": distances_matrix,
            "graph": G,
        }
        fadata.embedding_cache[embedding_graph_key] = embedding_basis_cache
    else:
        embedding_basis_cache = fadata.embedding_cache[embedding_graph_key]

    return embedding_basis_cache[return_type]


# TODO: add docs
def calculate_euclidean_distance_pc(fadata: FateAnnData, basis: str = None, model_name: str = None):
    # calculate correlation between euclidean distance(from specific cell) and pseudotime

    # extract from prior information
    if basis is None:
        basis = fadata.prior_information.get("basis")
    start_cell = fadata.prior_information["start_cell"]
    root_idx = fadata.obs.index.get_loc(start_cell)
    # pseudotime calculation
    pseudotime = fadata.get_trajectory_pseudotime(model_name=model_name)
    # distance matrix calculation
    # emb = fadata.obsm[basis][:, :2]
    # distance_array = emb - emb[root_idx]
    # euclidean_distance_array = np.sqrt(np.sum(distance_array**2, axis=1))
    euclidean_distance_array = get_embedding_graph(fadata, basis, return_type="distances_matrix")[root_idx].A
    # correlation calculation
    result = np.corrcoef(euclidean_distance_array, pseudotime)[1, 0]
    return result


# TODO: add docs
def calculate_geodesic_distance_pc(
    fadata: FateAnnData,
    basis: str = None,
    model_name: str = None,
    recompute_neighbors: bool = True,
    directed: bool = False,
):
    # calculate correlation between geodesic distance(from specific cell) and pseudotime

    # extract from prior information
    if basis is None:
        basis = fadata.prior_information.get("basis")
    start_cell = fadata.prior_information["start_cell"]
    root_idx = fadata.obs.index.get_loc(start_cell)

    # pseudotime calculation
    pseudotime = fadata.get_trajectory_pseudotime(model_name=model_name)

    # graph construction and get shortest path
    G = get_embedding_graph(
        fadata, basis, recompute_neighbors, directed, return_type="graph"
    )  # the embedding graph is available for various trajectory method
    shortest_paths_length_dict = nx.single_source_dijkstra_path_length(G, source=root_idx)  # path legth from start cell, sorted by length
    # set unreachable cells to max distance
    max_distance = max(shortest_paths_length_dict.values())
    for i in set(range(fadata.shape[0])) - set(shortest_paths_length_dict.keys()):
        shortest_paths_length_dict[i] = max_distance
        logger.debug(f"the {i} is unreachable , set to max distance({max_distance})")
    shortest_paths_length_dict = dict(sorted(shortest_paths_length_dict.items()))  # sort by raw cell order
    logger.debug(f"shortest_paths_length_dict : {shortest_paths_length_dict}")
    shortest_paths_length_array = np.array(list(shortest_paths_length_dict.values()))
    logger.debug(f"shortest_paths_length_array : {shortest_paths_length_array}")

    # correlation calculation
    result = np.corrcoef(shortest_paths_length_array, pseudotime)[1, 0]

    return result


# TODO: Remove redundent debug loggers
def calculate_recluster_f1(fadata: FateAnnData, basis: str = None, cluster: str = None, n_cluster=-1):
    # calculate clustering F1 between KMeans clustering and original clustering labels
    if basis is None:
        basis = fadata.prior_information.get("basis")
    if cluster is None:
        cluster = fadata.prior_information.get("cluster")
    emb = fadata.obsm[basis]
    if n_cluster == -1:
        # 默认聚类个数
        n_cluster = len(fadata.obs[cluster].cat.categories)
    logger.debug(f"n_cluster: {n_cluster}")

    # KMeans聚类
    logger.debug("Kmeans clustering...")
    model = KMeans(n_clusters=n_cluster, random_state=0).fit(emb)

    # F1值计算，参考论文PARC的Supplement 1与Github源代码
    logger.debug("Metric F1 calculating...")
    true_label_list = list(fadata.obs[cluster])
    pred_label_list = model.labels_

    logger.debug(f"true labels:{true_label_list}")
    logger.debug(f"pred labels:{pred_label_list}")
    f1 = calc_cluster_f1(true_label_list, pred_label_list)[0]
    return f1


def calc_cluster_f1(true_label_list, predict_label_list, merge_strategy="multiple_binary"):
    # 标号转换
    true_class_list = list(set(true_label_list))
    predict_class_list = list(set(predict_label_list))
    true_class_dict = dict([[i[1], i[0]] for i in enumerate(true_class_list)])
    predict_class_dict = dict([[i[1], i[0]] for i in enumerate(predict_class_list)])
    logger.debug(f"true_class_dict: \t{true_class_dict}")
    logger.debug(f"predict_class_dict: \t{predict_class_dict}")

    # 构造数量矩阵
    n_true_class = len(true_class_list)
    n_predict_class = len(predict_class_list)
    X = np.zeros((n_predict_class, n_true_class))
    for true_label, predict_label in zip(true_label_list, predict_label_list):
        i = true_class_dict[true_label]
        j = predict_class_dict[predict_label]
        X[j, i] += 1
    C_df = pd.DataFrame(np.array(X), index=predict_class_list, columns=true_class_list)
    # C_df = pd.DataFrame(X)
    # C_df.index = predict_class_list
    # C_df.columns = true_class_list
    logger.debug(f"C_df:\n {C_df}")

    if merge_strategy == "multiple_binary":
        # 转化为多个二分类问题
        predict2true = C_df.idxmax(axis=1)  # predict与true是多对一的关系
        logger.debug(f"predict2true: \n {predict2true}")
        n_label = len(true_label_list)  # 样本个数
        f1 = 0
        f1_weighted = 0
        for true_class in true_class_list:
            logger.debug(f"==========target is {true_class}==========")
            predict_class_list_sub = list(predict2true[predict2true == true_class].index)  # 对应的预测类
            for i in predict_class_list_sub:
                logger.debug(
                    f"==========cluster{i} has majority {true_class} with population {C_df.iloc[predict_class_dict[i]].sum()} , TP: {C_df.iloc[predict_class_dict[i], true_class_dict[true_class]]}"
                )
            logger.debug(f"{predict_class_list_sub}")
            true_label_list_masked = [1 if i == true_class else 0 for i in true_label_list]
            predict_label_list_masked = [1 if i in predict_class_list_sub else 0 for i in predict_label_list]
            logger.debug(f"true_label_list_masked: {true_label_list_masked}")
            logger.debug(f"predict_label_list_masked: {predict_label_list_masked}")
            f1_score_sub = f1_score(true_label_list_masked, predict_label_list_masked)
            logger.debug(f"==========f1_score_sub: {f1_score_sub}==========")
            f1 += f1_score_sub / n_true_class
            f1_weighted += f1_score_sub / n_label * sum(true_label_list_masked)

        return f1, f1_weighted
    else:
        # 实现与PARC论文中一致的指标, 匈牙利算法
        # ref: https://github.com/ShobiStassen/PARC
        P_df = C_df.apply(lambda x: x / x.sum(), axis=0)  # 列归一化
        R_df = C_df.apply(lambda x: x / x.sum(), axis=1)  # 行归一化
        F_df = 2 * (P_df * R_df).div(P_df + R_df).fillna(0)  # 安全除法，0做除数结果为0
        F_df.fillna(0)
        logger.debug(f"P_df:\n {P_df}")
        logger.debug(f"R_df:\n {R_df}")
        logger.debug(f"F_df:\n {F_df}")
        f1 = (C_df.sum(axis=0) * F_df.max(axis=0)).sum() / C_df.sum().sum()

        return f1


# ============================================================================================


# TODO: add docs
def calculate_cluster_silhouette(
    fadata: FateAnnData,
    basis: str = None,
    cluster: str = None,
):
    # calculate silhouette score based on clustering labels and embedding
    if basis is None:
        basis = fadata.prior_information.get("basis")
    if cluster is None:
        cluster = fadata.prior_information.get("cluster")

    # generate by Gemini 3 Pro, remove try-except for better error traceback
    # score = np.nan
    # if cluster and cluster in fadata.obs:
    #     labels = fadata.obs[cluster]
    #     # Silhouette requires at least 2 clusters
    #     if len(set(labels)) > 1:
    #         try:
    #             # 采样计算以提高速度 (如果数据量很大)
    #             if emb.shape[0] > 10000:
    #                 indices = np.random.choice(emb.shape[0], 10000, replace=False)
    #                 score = silhouette_score(emb[indices], labels[indices])
    #             else:
    #                 score = silhouette_score(emb, labels)
    #         except Exception as e:
    #             logger.warning(f"Failed to calculate silhouette score: {e}")
    #             score = np.nan
    #     else:
    #         score = np.nan
    score = silhouette_score(fadata.obsm[basis], fadata.obs[cluster])
    return score


# TODO: add docs
def calculate_striped_score(
    fadata: FateAnnData,
    basis: str = None,
    graph_type: str = "geodesic",
    # graph_type: str = "euclidean",
    use_weight: bool = False,
):
    # calculate strip score based on embedding
    # TODO: strip score is only valid for embedding with main structure, like UMAP, tSNE, PCA. Random embedding also get high strip score, need fixed.

    if basis is None:
        basis = fadata.prior_information.get("basis")
    emb = fadata.obsm[basis]

    # generate by Gemini 3 Pro, remove try-except for better error traceback
    # score = np.nan
    # try:
    #     # 计算 Embedding 空间及其 MST
    #     dist_mat = pairwise_distances(emb, emb)
    #     mst = minimum_spanning_tree(dist_mat)
    #     mst_graph = nx.Graph(mst)

    #     # 计算树的直径
    #     if nx.is_connected(mst_graph):
    #         diameter = nx.diameter(mst_graph)
    #     else:
    #         # 取最大连通分量
    #         largest_cc = max(nx.connected_components(mst_graph), key=len)
    #         subgraph = mst_graph.subgraph(largest_cc)
    #         diameter = nx.diameter(subgraph)

    #     n_nodes = emb.shape[0]
    #     # Normalized diameter [0, 1]
    #     score = diameter / (n_nodes - 1) if n_nodes > 1 else 0

    # except Exception as e:
    #     logger.warning(f"Failed to calculate elongation score: {e}")
    #     score = np.nan
    graph = get_embedding_graph(fadata, basis, graph_type=graph_type, return_type="graph")
    mst_graph = nx.minimum_spanning_tree(graph)

    # calculate tree diameter
    if nx.is_connected(mst_graph):
        # get diameter from connected graph directly
        diameter = nx.diameter(mst_graph, weight="weight" if use_weight else None)
    else:
        # get diameter from largest connected component
        largest_cc = max(nx.connected_components(mst_graph), key=len)
        subgraph = mst_graph.subgraph(largest_cc)
        diameter = nx.diameter(subgraph, weight="weight" if use_weight else None)

    n_nodes = emb.shape[0]
    score = diameter / (n_nodes - 1) if n_nodes > 1 else 0  # Normalized diameter [0, 1]

    return score


def plot_mst_diameter(
    fadata,
    basis="X_umap",
    graph_type: str = "geodesic",
    # graph_type: str = "euclidean",
    figsize=(8, 6),
    cluster_key="clusters",
    img_dir="figures",
):
    import matplotlib.pyplot as plt

    ax = plt.subplots(figsize=figsize)[1]
    # G = fadata.embedding_cache[f"{basis}_graph"]["G"]
    G = get_embedding_graph(fadata, basis, graph_type, return_type="graph")
    pos = fadata.obsm[basis][:, :2]
    nx.draw_networkx_edges(G=G, pos=pos, alpha=0.5, edge_color="gray", width=0.5)
    # MST
    G_mst = nx.minimum_spanning_tree(G)
    nx.draw_networkx_edges(G=G_mst, pos=pos, alpha=1, edge_color="black", width=1)
    # diameter path

    def get_diameter_path(tree):
        """获取树的直径路径 (Longest Shortest Path)"""
        # 处理非连通图的情况：取最大的连通分量
        if not nx.is_connected(tree):
            largest_cc = max(nx.connected_components(tree), key=len)
            tree = tree.subgraph(largest_cc)

        # 1. 第一次搜索：从任意点出发找最远点 u
        start_node = next(iter(tree.nodes()))
        lengths_from_start = nx.single_source_shortest_path_length(tree, start_node)
        u = max(lengths_from_start, key=lengths_from_start.get)

        # 2. 第二次搜索：从 u 出发找最远点 v
        lengths_from_u = nx.single_source_shortest_path_length(tree, u)
        v = max(lengths_from_u, key=lengths_from_u.get)

        # 3. 获取 u 到 v 的路径
        path = nx.shortest_path(tree, source=u, target=v)
        return path

    # --- 执行计算 ---
    diameter_path = get_diameter_path(G_mst)
    print(f"直径路径长度 (Edges): {len(diameter_path) - 1}")
    print(f"路径节点数: {len(diameter_path)}")

    # --- 可视化：在图上标出直径路径 (红色) ---
    # 假设 pos 已经在之前的代码定义了，例如 pos = fadata.obsm[basis]
    path_edges = list(zip(diameter_path, diameter_path[1:]))
    # 在原有的绘图基础上叠加红色路径
    nx.draw_networkx_edges(G=G_mst, pos=pos, edgelist=path_edges, edge_color="red", width=2)
    sc.pl.embedding(fadata, color=cluster_key, basis=basis, frameon=False, show=False, legend_loc=None, title=basis, ax=ax, size=15)
    plt.savefig(f"{img_dir}/mst_diameter_{basis}.pdf")
    plt.show()


def calculate_embedding_metric(
    fadata: FateAnnData,
    basis: str = None,
    model_name: str = None,
    pre_trajectory: bool = True,
    post_trajectory: bool = True,
):
    metric = {}
    if pre_trajectory:
        metric["recluster_f1"] = calculate_recluster_f1(fadata, basis)
        metric["cluster_silhouette"] = calculate_cluster_silhouette(fadata, basis)
        metric["striped_score"] = calculate_striped_score(fadata, basis)
    if post_trajectory:
        metric["euclidean_distance_pc"] = calculate_euclidean_distance_pc(fadata, basis, model_name)
        metric["geodesic_distance_pc"] = calculate_geodesic_distance_pc(fadata, basis, model_name)
    return metric

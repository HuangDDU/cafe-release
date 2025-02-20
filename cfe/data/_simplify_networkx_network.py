import pandas as pd
import networkx as nx


def simplify_networkx_network(
    gr: nx.Graph,
    allow_duplicated_edges=True,
    allow_self_loops=True,
    force_keep=[],
    edge_points=None
):
    # 基于NetworkX简化milestone network

    # TODO: 无向图的边可以互相反转
    if edge_points is not None:
        edge_points = edge_points

    # 重新命名确保不混乱
    gr = nx.relabel_nodes(gr, {name: f"#M#{name}" for name in gr.nodes})
    if edge_points is not None:
        edge_points["from"] = edge_points["from"].apply(lambda x: f"#M#{x}")
        edge_points["to"] = edge_points["to"].apply(lambda x: f"#M#{x}")
    if force_keep is not None:
        force_keep = [f"#M#{name}" for name in force_keep]

    # 边权没有则手动添加为1
    attribute_keys = list(gr.edges(data=True))[0][2].keys()  # 提取第一个边的属性键名， 所有边的属性键名一致
    if "weight" not in attribute_keys:
        for u, v in gr.edges():
            gr[u][v]["weight"] = 1

    # 边是否有方向未指定，则手动添加与gr保持一致
    is_directed = gr.is_directed()
    if "directed" not in attribute_keys:
        for u, v in gr.edges():
            gr[u][v]["directed"] = is_directed

    # 分别处理每个连通分量
    if is_directed:
        connected_component_list = list(nx.weakly_connected_components(gr))  # 有向图提取弱连通分量，连接即可
    else:
        connected_component_list = list(nx.connected_components(gr))
    simplified_graphs = []
    for connected_component in connected_component_list:
        subgr = gr.subgraph(connected_component).copy()
        simplified_subgraph = simplify_subgraph(subgr, is_directed, force_keep, edge_points)
        simplified_graphs.append(simplified_subgraph)

    # 合并图，这里直接合并
    out_gr = nx.compose_all([i["subgr"] for i in simplified_graphs])
    seps = [i["sub_edge_points"] for i in simplified_graphs]

    out_gr = nx.relabel_nodes(out_gr, {name: name[3:] for name in out_gr.nodes})  # 名字改回去

    if edge_points is None:
        return out_gr  # 暂时只输出简化后的networkx图结构
    else:
        seps = pd.concat(seps)
        seps["from"] = seps["from"].apply(lambda x: x[3:])
        seps["to"] = seps["to"].apply(lambda x: x[3:])
        return {
            "gr": out_gr,
            "edge_points": seps
        }


def simplify_subgraph(subgr, is_directed, force_keep, edge_points):
    # NOTE: 这里是简化的核心函数
    node_list = list(subgr.nodes)
    keep_v = simplify_determine_nodes_to_keep(subgr, is_directed, force_keep)  # 决定保留哪些节点True， 过滤哪些节点False
    if edge_points is not None:
        edge = pd.DataFrame(data=subgr.edges(), columns=["from", "to"])
        edge_rev = edge.rename(columns={"from": "to", "to": "from"}).drop_duplicates()
        edge_bothdir = pd.concat([edge, edge_rev], axis=0)  # 双向边添加，与前面的无向图边反转一致
        sub_edge_points = pd.merge(edge_points, edge_bothdir, on=["from", "to"])
    else:
        sub_edge_points = None
    keep_v = keep_v  # TODO:
    num_vs = len(subgr.nodes)
    neighs = simplify_get_neighbours(subgr, is_directed)
    to_process = [not i for i in keep_v]
    for v_rem in range(num_vs):
        # 从特定位置开始，向前（入度）向后（出度）搜索删除链
        if to_process[v_rem]:
            to_process[v_rem] = False
            # 向前、入度前驱节点、边处理
            i = simplify_get_i(neighs, v_rem, is_directed)  # 前驱节点
            i_prev = v_rem
            left_path = [{"from": i, "to": i_prev, "weight": simplify_get_edge(subgr, i, i_prev)["weight"]}]
            while to_process[i]:
                # 前驱节点仍然需要删除
                to_process[i] = False
                tmp = i
                i = simplify_get_next(neighs, i, is_directed, left=True, prev=i_prev)
                i_prev = tmp
                # left_path.append({"from": i, "to": i_prev, "weight": simplify_get_edge(subgr, i, i_prev)["weight"]})
                left_path.append({"from": i, "to": i_prev, "weight": subgr.edges[(node_list[i], node_list[i_prev])]["weight"]})

            # 向后、出度后继节点、边处理
            j = simplify_get_j(neighs, v_rem, is_directed)  # 后继节点
            j_prev = v_rem
            right_path = [{"from": j_prev, "to": j, "weight": simplify_get_edge(subgr, j_prev, j)["weight"]}]
            while to_process[j]:
                # 后继节点仍然需要删除
                to_process[j] = False
                tmp = j
                j = simplify_get_next(neighs, j, is_directed, left=False, prev=j_prev)
                j_prev = tmp
                # right_path.append({"from": j_prev, "to": j, "weight": simplify_get_edge(subgr, j_prev, j)["weight"]})
                right_path.append({"from": j_prev, "to": j, "weight": subgr.edges[(node_list[j_prev], node_list[j])]["weight"]})

            # 拼接后，节点序号转换为节点名字
            left_path = pd.DataFrame(left_path).iloc[::-1].reset_index(drop=True) # 前驱查找需要翻转顺序
            left_path["from"] = [node_list[i] for i in left_path["from"]]
            left_path["to"] = [node_list[i] for i in left_path["to"]]
            right_path = pd.DataFrame(right_path)
            right_path["from"] = [node_list[i] for i in right_path["from"]]
            right_path["to"] = [node_list[i] for i in right_path["to"]]

            # 此时i,j为删除链的前驱后继序号
            if i == j:
                # TODO: 自环等操作
                pass
            else:
                rplcd = simplify_replace_edges(subgr, sub_edge_points, i, j, path=pd.concat([left_path, right_path]), is_directed=is_directed)
                subgr = rplcd["subgr"]
                sub_edge_points = rplcd["sub_edge_points"]
    subgr.remove_nodes_from(list(keep_v[~keep_v].index))
    return {
        "subgr": subgr,
        "sub_edge_points": sub_edge_points
    }


def simplify_determine_nodes_to_keep(subgr: nx.Graph | nx.DiGraph, is_directed, force_keep=[]):
    # 决定在简化过程中保留哪些节点
    node_list = list(subgr.nodes)
    name_check = pd.Series([i in force_keep for i in node_list], index=node_list)
    loop_check = pd.Series(False, index=node_list)
    loop_check.loc[nx.nodes_with_selfloops(subgr)] = True
    degr_check = []
    if is_directed:
        # 有向图过滤入度1且出度1的节点
        in_degrees = subgr.in_degree
        out_degrees = subgr.out_degree
        for node in node_list:
            in_deg = in_degrees[node]
            out_deg = out_degrees[node]
            if in_deg == 1 and out_deg == 1:
                degr_check.append(False)
            else:
                degr_check.append(True)
    else:
        # 无向图过滤度为2的节点
        degrees = subgr.degree
        for node in node_list:
            deg = degrees[node]
            if deg == 2:
                degr_check.append(False)
            else:
                degr_check.append(True)

    return name_check | loop_check | degr_check  # 或运算符，满足一个条件就保留


def simplify_get_neighbours(subgr, is_directed):
    #  获得图上所有节点的邻居节点序号
    name2id = {name: i for i, name in enumerate(subgr.nodes)}  # name -> id(0,1,2...)
    neighs = {}
    if is_directed:
        neighs["neighs_in"] = []
        neighs["neighs_out"] = []
    else:
        neighs["neighs"] = []
    for node in subgr.nodes:
        if is_directed:
            in_neighbors = [name2id[i] for i in list(subgr.predecessors(node))]  # 前继
            out_neighbors = [name2id[i] for i in list(subgr.successors(node))]  # 后驱
            neighs["neighs_in"] .append(in_neighbors)
            neighs["neighs_out"].append(out_neighbors)
        else:
            neighs["neighs"].append([name2id[i] for i in list(subgr.neighbors(node))])
    return neighs


def simplify_get_i(neighs, v_rem, is_directed):
    # 获取入度节点
    if is_directed:
        return neighs["neighs_in"][v_rem][0]  # 暂时认为只有一个前驱
    else:
        return neighs["neighs"][v_rem][0]


def simplify_get_j(neighs, v_rem, is_directed):
    # 获取出度节点
    if is_directed:
        return neighs["neighs_out"][v_rem][0]  # 暂时认为只有一个后继
    else:
        return neighs["neighs"][v_rem][1]


def simplify_get_next(neighs, v_rem, is_directed, left=False, prev=None):
    # 连续删除节点，prev是上一次删除的节点，v_rem是当前删除的节点
    if is_directed:
        if left:
            # 有向图上不会产生与prev重复的节点
            return neighs["neighs_in"][v_rem][0]  # 入度即向前
        else:
            return neighs["neighs_out"][v_rem][0]  # 出度即向后
    else:
        return list(set(neighs["neighs"][v_rem]) - set([prev]))[0] # 无向图上要除去上一步的节点


def anti_join(df_left, df_right, on=None):
    # 反连接，只在df_left中出现的行，模拟 dplyr 中的 anti_join 操作
    merged_df = df_left.merge(df_right, on=on, how="left", indicator=True, suffixes=("", "_y"))
    return merged_df[merged_df["_merge"] == "left_only"].drop(columns="_merge")[df_left.columns.tolist()]


def simplify_get_edge_points_on_path(
        sub_edge_points: pd.DataFrame,
        path: pd.DataFrame
        ):
    """获得在子图milestone_percentage待删除的路径的细胞

    Args:
        sub_edge_points (pd.DataFrame): 子图milestone_percentage
        path (pd.DataFrame): 待删除的路径

    Returns:
        dict: _description_
    """
    # 对于边上细胞的的处理
    rev_path = path.rename(columns={"from": "to", "to": "from"})[["from", "to"]] # 边反转

    # 拼接反转后的边和sub_edge_point
    sepaj = anti_join(sub_edge_points, path, on=["from", "to"]) # 仅仅在sub_edge_points但不在path的细胞保留
    tofilp = pd.merge(sepaj, rev_path, on=["from", "to"]) # 反转细胞：sepaj中还有细胞也应该被剔除掉，这些细胞在rev_path中的边可以在sepaj中找到

    tofilp_tmp = tofilp.rename(columns={"from": "to", "to": "from"})
    tofilp_tmp["percentage"] = 1 - tofilp_tmp["percentage"] # 反转细胞percentage计算并额外拼接上去
    both_sub_edge_points = pd.concat([sub_edge_points, tofilp_tmp]) 
    on_path = pd.merge(both_sub_edge_points, path, on=["from", "to"]) # 在待删除边上的细胞

    not_on_path = anti_join(sepaj, rev_path, on=["from", "to"]) # 不在带删除边上的细胞

    return {
        "on_path": on_path,
        "not_on_path": not_on_path
    }


def simplify_replace_edges(subgr: nx.Graph | nx.DiGraph, sub_edge_points, i, j, path, is_directed):
    # 添加替换旧边的新边
    node_list = list(subgr.nodes)

    swap = (not is_directed) and (i > j)
    if swap:
        i, j = j, i

    path_len = path["weight"].sum()
    subgr.add_edge(node_list[i], node_list[j], weight=path_len, directed=is_directed)  # 添加新边
    if sub_edge_points is not None:
        # 边上的点处理
        path["cs"] = path["weight"].cumsum() - path["weight"]  # 从dataframe表头开始的累计边长
        out = simplify_get_edge_points_on_path(sub_edge_points, path)
        processed_edge_points = out["on_path"]
        processed_edge_points["from"] = node_list[i]
        processed_edge_points["to"] = node_list[j]
        if not path_len == 0:
            # 关键部分，修改percentage
            cs = processed_edge_points["cs"]
            percentage = processed_edge_points["percentage"]
            weight = processed_edge_points["weight"]
            processed_edge_points["percentage"] = (cs + percentage * weight) / path_len
            processed_edge_points = processed_edge_points[["id", "from", "to", "percentage"]]
        else:
            processed_edge_points["percentage"] = 0.5
        if swap:
            processed_edge_points["percentage"] = 1 - processed_edge_points["percentage"]

        sub_edge_points = pd.concat([out["not_on_path"], processed_edge_points])
    return {
        "subgr": subgr,
        "sub_edge_points": sub_edge_points
    }


def simplify_get_edge(subgr, i, j):
    node_list = list(subgr.nodes)
    return subgr.edges[(node_list[i], node_list[j])]

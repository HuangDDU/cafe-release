import networkx as nx
import pandas as pd

from .._logging import logger


def simplify_networkx_network(
    gr: nx.Graph,
    # allow_duplicated_edges=True,
    # allow_self_loops=True,
    force_keep=[],
    edge_points=None,
    prune_threshold=0.0,
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

    # ------------------------------------------------------------------
    # 1. 计算剪枝的绝对阈值 (Strategy)
    # ------------------------------------------------------------------
    if prune_threshold > 0:
        # 使用无向图逻辑来判断分支长度（即使是有向图，物理结构上的分支也是无向的）
        undirected_gr = gr.to_undirected() if is_directed else gr
        # 计算实际阈值
        if prune_threshold < 1.0:
            try:
                # 尝试计算直径（最长路径），如果图不连通则取最大连通分量的直径
                if nx.is_connected(undirected_gr):
                    diameter = nx.diameter(undirected_gr, weight="weight")
                else:
                    largest_cc = max(nx.connected_components(undirected_gr), key=len)
                    diameter = nx.diameter(undirected_gr.subgraph(largest_cc), weight="weight")
                threshold = diameter * prune_threshold
            except Exception:
                # 如果计算直径失败（例如图太大或有孤立点），回退到最大边权 * 10
                max_weight = max([d.get("weight", 1) for u, v, d in undirected_gr.edges(data=True)], default=1)
                threshold = max_weight * 10
        else:
            threshold = prune_threshold
    else:
        threshold = 0.0
    logger.debug(f"applying pruning with threshold {threshold}...")

    # ------------------------------------------------------------------
    # 2. 分别处理每个连通分量 (Execution)
    # ------------------------------------------------------------------
    if is_directed:
        connected_component_list = list(nx.weakly_connected_components(gr))  # 有向图提取弱连通分量，连接即可
    else:
        connected_component_list = list(nx.connected_components(gr))
    logger.debug(f"simplifying {len(connected_component_list)} connected components...")
    simplified_graphs = []
    for connected_component in connected_component_list:
        subgr = gr.subgraph(connected_component).copy()
        diameter = nx.diameter(subgr, weight="weight")
        if diameter < threshold:
            logger.debug(f"remove small component, diameter {diameter:.2f}, size:{len(subgr.nodes)}, nodes:{subgr.nodes}")
            # skip small components
            continue
        simplified_subgraph = simplify_subgraph(subgr, is_directed, force_keep, edge_points, threshold)
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
        return {"gr": out_gr, "edge_points": seps}


def simplify_subgraph(subgr, is_directed, force_keep, edge_points, threshold=0.0):
    # NOTE: 这里是简化的核心函数

    # 1. 准备 edge_points (提取属于当前子图的边上的点)
    if edge_points is not None:
        edge = pd.DataFrame(data=subgr.edges(), columns=["from", "to"])
        edge_rev = edge.rename(columns={"from": "to", "to": "from"}).drop_duplicates()
        edge_bothdir = pd.concat([edge, edge_rev], axis=0)  # 双向边添加，与前面的无向图边反转一致
        sub_edge_points = pd.merge(edge_points, edge_bothdir, on=["from", "to"])
    else:
        sub_edge_points = None
    # ------------------------------------------------------------------
    # [新增] Step 1: 剪枝 (Pruning)
    # ------------------------------------------------------------------
    if threshold > 0:
        # 使用无向逻辑判断分支
        undirected_subgr = subgr.to_undirected() if is_directed else subgr
        prune_map = _identify_prunable_nodes_map(undirected_subgr, threshold, force_keep)

        if prune_map:
            logger.debug(f"Subgraph(size:{subgr.size()}) pruning: removing {len(prune_map)} nodes.")
            # A. 调整细胞位置 (Snap to backbone)
            if sub_edge_points is not None:
                sub_edge_points = _snap_points_to_backbone(sub_edge_points, prune_map)

            # B. 物理删除节点
            subgr.remove_nodes_from(prune_map.keys())

    # ------------------------------------------------------------------
    # Step 2: 链式简化 (Chain Simplification)
    # ------------------------------------------------------------------
    # 注意：剪枝后图结构变了，必须重新计算哪些节点需要保留
    node_list = list(subgr.nodes)
    keep_v = simplify_determine_nodes_to_keep(subgr, is_directed, force_keep)  # 决定保留哪些节点True，过滤哪些节点False

    num_vs = len(subgr.nodes)
    neighs = simplify_get_neighbours(subgr, is_directed)
    to_process = (~keep_v).tolist()
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
            left_path = pd.DataFrame(left_path).iloc[::-1].reset_index(drop=True)  # 前驱查找需要翻转顺序
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
    return {"subgr": subgr, "sub_edge_points": sub_edge_points}


def _identify_prunable_nodes_map(G, threshold, force_keep):
    """
    识别需要剪枝的节点，并记录它们应该“坍缩”到哪个骨架节点上。

    该算法采用迭代策略，从叶子节点向内搜索，直到遇到分叉点（骨架）或超过长度阈值。

    Args:
        G (nx.Graph): 待处理的图（通常是无向图）。
        threshold (float): 剪枝的长度阈值。分支总长度小于此值的将被剪除。
        force_keep (list): 强制保留的节点列表。

    Returns:
        dict: 映射字典 {被删除的节点: 吸附的目标骨架节点}。
    """
    # generated by Gemini
    prune_map = {}

    # 迭代循环：因为剪掉一层叶子后，原本的内部节点可能变成新的叶子（多级分支），
    # 所以需要反复检查，直到没有新的节点被标记为删除。
    while True:
        current_round_map = {}

        # 1. 寻找当前的叶子节点（度为1）
        # 排除掉已经被标记删除的节点，以及强制保留的节点
        leaves = [n for n, d in G.degree() if d == 1 and n not in prune_map and n not in force_keep]

        if not leaves:
            break  # 没有叶子了，结束

        for leaf in leaves:
            path_nodes = [leaf]  # 当前分支上的节点列表
            curr = leaf
            dist = 0
            keep_branch = True  # 默认保留，除非满足剪枝条件
            snap_target = None  # 如果剪枝，这些节点将吸附到哪个骨架节点上

            # 2. 从叶子向内回溯，测量分支长度
            while True:
                # 获取邻居（排除已标记删除的节点和当前路径上的节点）
                neighbors = [n for n in G.neighbors(curr) if n not in prune_map and n not in path_nodes]
                if not neighbors:
                    break  # 孤立点或死胡同

                parent = neighbors[0]  # 在树状分支上，向内只有一个父节点
                edge_len = G[curr][parent].get("weight", 1)
                dist += edge_len

                # A. 如果分支长度超过阈值，则保留该分支
                if dist > threshold:
                    keep_branch = True
                    break

                # 计算父节点的“有效度数”（忽略已标记删除的邻居）
                effective_degree = len([n for n in G.neighbors(parent) if n not in prune_map])

                # B. 遇到分叉点 (有效度数 > 2)
                # 说明我们到达了骨架（Backbone），且距离在阈值内 -> 剪掉这个分支
                if effective_degree > 2:
                    keep_branch = False
                    snap_target = parent  # 分支上的点都吸附到这个分叉点
                    break

                # C. 遇到另一个叶子 (有效度数 == 1)
                # 说明这是一条很短的孤立线段（两头都是叶子），通常保留以防删空
                elif effective_degree == 1:
                    keep_branch = True
                    break

                # D. 遇到线性中间点 (有效度数 == 2)
                # 继续沿路径向内搜索
                path_nodes.append(parent)
                curr = parent

            # 3. 如果判定为剪枝，记录映射关系
            if not keep_branch and snap_target is not None:
                for node in path_nodes:
                    current_round_map[node] = snap_target

        # 如果本轮没有发现可剪枝的节点，则退出循环
        if not current_round_map:
            break

        # 更新总的剪枝映射表
        prune_map.update(current_round_map)

    return prune_map


def _snap_points_to_backbone(edge_points, prune_map):
    """将位于被删除节点上的细胞移动到最近的骨架节点上"""
    df = edge_points.copy()
    df["from"] = df["from"].apply(lambda x: prune_map.get(x, x))
    df["to"] = df["to"].apply(lambda x: prune_map.get(x, x))
    return df


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
            neighs["neighs_in"].append(in_neighbors)
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
        return list(set(neighs["neighs"][v_rem]) - set([prev]))[0]  # 无向图上要除去上一步的节点


def anti_join(df_left, df_right, on=None):
    # 反连接，只在df_left中出现的行，模拟 dplyr 中的 anti_join 操作
    merged_df = df_left.merge(df_right, on=on, how="left", indicator=True, suffixes=("", "_y"))
    return merged_df[merged_df["_merge"] == "left_only"].drop(columns="_merge")[df_left.columns.tolist()]


def simplify_get_edge_points_on_path(sub_edge_points: pd.DataFrame, path: pd.DataFrame):
    """获得在子图milestone_percentage待删除的路径的细胞

    Args:
        sub_edge_points (pd.DataFrame): 子图milestone_percentage
        path (pd.DataFrame): 待删除的路径 (包含 from, to, weight, cs)

    Returns:
        dict: {"on_path": DataFrame, "not_on_path": DataFrame}
    """
    # 1. 构造完整的路径边集合（正向 + 反向）
    # path 包含待合并链上的所有边: A->B, B->C ...
    path_edges = path[["from", "to"]].copy()
    path_edges["_is_forward"] = True

    rev_path_edges = path_edges.rename(columns={"from": "to", "to": "from"})
    rev_path_edges["_is_forward"] = False

    # all_path_edges 包含了链上所有的边，用于匹配细胞
    all_path_edges = pd.concat([path_edges, rev_path_edges])

    # 2. 将细胞数据与路径边进行匹配
    # 使用 left join 保留所有细胞，通过 _merge 或指示列判断是否在路径上
    merged = pd.merge(sub_edge_points, all_path_edges, on=["from", "to"], how="left")

    # 3. 分离 "在路径上" 和 "不在路径上" 的细胞
    mask_on_path = merged["_is_forward"].notna()

    on_path_raw = merged[mask_on_path].copy()
    not_on_path = merged[~mask_on_path].drop(columns=["_is_forward"])

    # 4. 统一 "在路径上" 的细胞方向
    # 如果细胞匹配到的是反向边 (_is_forward == False)，需要翻转 from/to 并重算 percentage
    # 注意：这里的翻转是为了让它们跟 path 的方向一致，方便后续计算累积距离

    # 找出需要翻转的行
    mask_rev = ~on_path_raw["_is_forward"]

    if mask_rev.any():
        # 翻转 from/to
        on_path_raw.loc[mask_rev, ["from", "to"]] = on_path_raw.loc[mask_rev, ["to", "from"]].values
        # 翻转 percentage (1 - p)
        on_path_raw.loc[mask_rev, "percentage"] = 1.0 - on_path_raw.loc[mask_rev, "percentage"]

    # 5. 将 path 的元数据 (weight, cs) merge 进去，以便计算新位置
    # 此时 on_path_raw 的 from/to 已经全部调整为正向，可以直接与 path merge
    on_path = pd.merge(on_path_raw.drop(columns=["_is_forward"]), path, on=["from", "to"], how="left")

    return {"on_path": on_path, "not_on_path": not_on_path}


def simplify_replace_edges(subgr: nx.Graph | nx.DiGraph, sub_edge_points, i, j, path, is_directed):
    # 添加替换旧边的新边
    node_list = list(subgr.nodes)

    # 确定新边的起点和终点索引
    u_idx, v_idx = i, j

    # 如果是无向图，通常保持 ID 小的在前，或者由调用者保证顺序
    # 原代码逻辑：swap = (not is_directed) and (i > j)
    # 但注意：path 的方向是从 i 到 j 的。如果 swap 了，path 的累积距离计算方向也得反过来。
    # 为了简化，我们先按 i->j 计算，最后如果需要 swap 再统一翻转结果。

    swap_final_result = (not is_directed) and (i > j)
    if swap_final_result:
        u_idx, v_idx = j, i

    # 计算新边总长
    path_len = path["weight"].sum()

    # 添加新边 (u -> v)
    subgr.add_edge(node_list[u_idx], node_list[v_idx], weight=path_len, directed=is_directed)

    if sub_edge_points is not None:
        # 预计算路径上每段的累积起始距离 (Cumulative Sum)
        # cs 表示当前边的起点距离 path 起点 (i) 的距离
        # 比如 path: A--1-->B--2-->C
        # A->B: weight=1, cs=0
        # B->C: weight=2, cs=1
        path = path.copy()  # 避免修改外部变量
        path["cs"] = path["weight"].cumsum() - path["weight"]

        # 获取并处理细胞
        out = simplify_get_edge_points_on_path(sub_edge_points, path)

        processed_edge_points = out["on_path"]

        # 将所有这些细胞映射到新边 i -> j 上
        # 此时 processed_edge_points 里的 percentage 是相对于各自小边的
        # cs 是小边起点相对于 i 的距离
        # weight 是小边的长度

        if path_len > 1e-9:  # 避免除以0
            # 新位置 = (小边起点距离 + 细胞在小边上的相对距离 * 小边长度) / 总长度
            processed_edge_points["percentage"] = (
                processed_edge_points["cs"] + processed_edge_points["percentage"] * processed_edge_points["weight"]
            ) / path_len
        else:
            processed_edge_points["percentage"] = 0.5

        # 设置新边的 from/to (暂时设为 i -> j)
        processed_edge_points["from"] = node_list[i]
        processed_edge_points["to"] = node_list[j]

        # 如果最终决定新边是 j -> i (swap_final_result)，则需要翻转
        if swap_final_result:
            processed_edge_points["from"] = node_list[j]
            processed_edge_points["to"] = node_list[i]
            processed_edge_points["percentage"] = 1.0 - processed_edge_points["percentage"]

        # 只保留必要的列
        cols = ["id", "from", "to", "percentage"]
        # 如果 sub_edge_points 有其他列也应该保留，这里假设只处理这几列，或者取交集
        existing_cols = [c for c in cols if c in processed_edge_points.columns]
        processed_edge_points = processed_edge_points[existing_cols]

        # 合并结果
        sub_edge_points = pd.concat([out["not_on_path"], processed_edge_points], ignore_index=True)

    return {"subgr": subgr, "sub_edge_points": sub_edge_points}


def simplify_get_edge(subgr, i, j):
    node_list = list(subgr.nodes)
    return subgr.edges[(node_list[i], node_list[j])]

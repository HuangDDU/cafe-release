from itertools import combinations, product

import numpy as np
import pandas as pd
import networkx as nx
from ...data import simplify_networkx_network


def calculate_edge_flip(
    net1: pd.DataFrame,
    net2: pd.DataFrame,
    return_type="score",
    simplify=False, # 提前简化过了
    limit_flips=5,
    limit_combinations=12650
):
    # get the matched adjacencies
    # 提取邻接矩阵，忽略边权
    adjacencies = get_matched_adjacencies(net1, net2, simplify=simplify)
    adj1 = adjacencies[0] > 0
    adj2 = adjacencies[1] > 0

    # calculate the mapping which nodes are connected to which edges
    # 计算节点与边的对应关系，行为节点，列为边。这里为所有的边有C_n^2种排列，n为节点数量
    edge_membership1 = calculate_edge_membership(adj1)

    # substract the number of edges
    # 提取下三角矩阵，计算边的数量差异
    adj1_tril_mask = np.tril(adj1.values, k=-1)  # 去除对角线的下三角的邻接矩阵
    adj2_tril_mask = np.tril(adj2.values, k=-1)
    edge_difference =  adj2_tril_mask.sum() - adj1_tril_mask.sum()

    # calculate the possible edges which can be added and removed to net1
    # 计算可能的添加、删除边再edge_membership1中的序号
    possible_edge_additions = []
    possible_edge_removes = []
    # 模拟下三角序号计算
    index2edge = edge_membership1.apply(lambda x: tuple(x.index[x == 1]), axis=1).to_dict()
    edge2index = {edge: index for index, edge in index2edge.items()}
    nodes1 = adj1.index.tolist()
    n_adj1 = adj1.shape[0]
    for j in range(n_adj1):
        for i in range(j+1, n_adj1):
            index = edge2index[(nodes1[j], nodes1[i])]
            if adj1_tril_mask[i, j]:
                possible_edge_removes.append(index)
            else:
                possible_edge_additions.append(index)

    G2 = nx.from_pandas_adjacency(adj2)  # used later to calculate isomorphism # 使用后者net2计算同构，即net2不变，一直对net1调整
    sorted_degrees2 = adj2.sum().sort_values().values  # used later to compare degree distributions # 对net2节点进行度排序用作后续对比

    # prepare for looping over the number of edges which can be flipped
    # 准备循环
    found = False
    n_flips = abs(edge_difference) - 2

    # determine upper bound
    # 启发式搜索终止条件，最大反转次数
    upper_bound = adj1_tril_mask.sum() + adj2_tril_mask.sum() - 2
    if upper_bound <= 0:
        upper_bound = 1

    while (not found) and (n_flips <= upper_bound):
        n_flips += 2

        if n_flips > limit_flips:
            # 超过了限定的最多的边反转次数，算是没有找到
            n_flips = upper_bound
            break
        else:
            # calculate the number of additions and removes
            # 计算添加和删除边的数量
            n_additions = int((n_flips + edge_difference)/2)
            n_removes = int((n_flips - edge_difference)/2)
            if n_additions < 0 or n_removes < 0:
                raise "Edge additions and removes should be integer and higher than 0"
            else:
                if (len(list(combinations(possible_edge_additions, n_additions))) > limit_combinations) or (len(list(combinations(possible_edge_removes, n_removes))) > limit_combinations):
                    n_flips = upper_bound
                    break
                else:
                    # create the matrix which contains in the columns all possible flips, with in the rows the edge_id which will be flipped
                    # 创建矩阵，在列表中包含所有的可能反转，在行中包含将翻转的edge_id
                    edge_additions = [list(i) for i in combinations(possible_edge_additions, n_additions)]
                    edge_removes = [list(i) for i in combinations(possible_edge_removes, n_removes)]

                    if n_additions > 0 and n_removes > 0:
                        edge_flips = [i[0]+i[1] for i in product(edge_additions, edge_removes)]  # 添加和删除组合，相当于是笛卡尔积
                    elif n_additions > 0:
                        edge_flips = edge_additions
                    else:
                        edge_flips = edge_removes

                    edge_flips = np.array(edge_flips).T

                    # cut the edge_flips, avoiding huge memory consumption
                    # 分成多组来计算，避免后续巨大的内存消耗
                    grouping = np.arange(edge_flips.shape[1]) // 1000
                    ngroups = max(grouping)
                    group_id = -1  # 后续从0开始

                    # loop over each group of edge_flips
                    # 每一组分别批量计算
                    while (not found) and (group_id < ngroups):
                        group_id = group_id + 1
                        edge_flips_group = edge_flips[:, grouping == group_id]

                        # generate matrix with in the columns each flip and in the rows the vector format of the new adjacency of net1
                        # 生成矩阵的1列为1种flip翻转后的邻接矩阵的向量格式
                        edge_flip_vectors1 = generate_edge_flip_vectors(edge_flips_group, adj1, possible_edge_removes)  # (n_flips, n_edge) # 这里额外给要已有的边
                        degree_vectors1 = edge_flip_vectors1 @ edge_membership1.values  # (n_flips, n_nodes)

                        # now check several metrics of the new adjacency matrix, from fastest to slowest
                        # after each check, the flips which are not OK are removed (in the selected object)
                        # 从快到慢地检查几个指标，保留通过地指标用作后续图同构判断
                        selected = np.arange(degree_vectors1.shape[0])

                        # 快速最大度检查
                        degree_max_check = check_degrees_max(degree_vectors1[selected], sorted_degrees2)
                        if degree_max_check.any():
                            selected = selected[degree_max_check]

                            # 快速最小度检查
                            degree_min_check = check_degrees_min(degree_vectors1[selected], sorted_degrees2)
                            if degree_min_check.any():
                                selected = selected[degree_min_check]

                                # 度排序检查
                                degree_sorted_check = check_degrees_sorted(degree_vectors1[selected], sorted_degrees2)
                                if degree_sorted_check.any():
                                    selected = selected[degree_sorted_check]

                                    # 上述一堆度排序是为了避免不必要的图同构检测，现在可以检测同构了
                                    for edge_flip in edge_flips_group[:, selected].T:
                                        new_adj1 = flip_adj(edge_flip, adj1, index2edge)
                                        G1_new = nx.from_pandas_adjacency(new_adj1, create_using=nx.Graph)
                                        if nx.is_isomorphic(G1_new, G2):
                                            found = True
                                            new_adj1 = new_adj1
                                            break

    score = 1 - n_flips / upper_bound
    if return_type == "score":
        return score

    else:
        return {
            "score": score,
            "newadj1": new_adj1,
            "oldadj1": adj1
        }


def get_adjacency_lengths(net, nodes=None):
    # 提取邻接矩阵
    if nodes is None:
        nodes = sorted(net[["from", "to"]].stack().unique())
    if net.shape[0] == 0:
        # 没有边，全部是离散结点
        newnet = pd.DataFrame(0, index=nodes, columns=nodes)
    else:
        # 转化为邻接矩阵
        net["from"] = pd.Categorical(net["from"], categories=nodes)
        net["to"] = pd.Categorical(net["to"], categories=nodes)
        newnet = net.pivot_table(index="from", columns="to", values="length", aggfunc="sum", fill_value=0)
        newnet = newnet.reindex(index=nodes, columns=nodes, fill_value=0)
    return newnet + newnet.T


def complete_matrix(mat, dim, fill=0):
    # add extra rows and columns to matrix
    # 添加额外的行列以补全矩阵
    old_dim = mat.shape[0]
    new_mat = np.zeros((dim, dim))
    new_mat[:old_dim, :old_dim] = mat.values
    nodes = mat.index.tolist() + list(range(dim-old_dim))
    new_mat = pd.DataFrame(new_mat, index=nodes, columns=nodes)
    return new_mat


def get_matched_adjacencies(net1, net2, simplify=True,):
    # 获得简化处理后的邻接矩阵

    if simplify:
        def simplify_net(net):
            # 转化为networkX对象进行简化
            directed = net["directed"].any()
            net = net.rename(columns={"length": "weight"})
            net = net.query("`from`!=`to` and `weight`!=0")  # 去除自环边和长度为0的边
            G = nx.from_pandas_edgelist(net, source="from", target="to", create_using=nx.Graph)  # 创建无向图
            G = simplify_networkx_network(G)  # 简化图 # TODO: 简化图的策略需要调整
            net = nx.to_pandas_edgelist(G)
            net = net.rename(columns={"weight": "length", "source": "from", "target": "to"})
            net["directed"] = directed
            # TODO: 暂时不考虑自环和重复边
            return net

        net1 = simplify_net(net1)
        net2 = simplify_net(net2)

    # 获得邻接矩阵，返回结果为DataFrame
    adj1 = get_adjacency_lengths(net1)
    adj2 = get_adjacency_lengths(net2)

    # make the adjacency matrices have the same dimensions
    # 补充使其具有相同的维度
    if (adj1.shape[0] >= adj2.shape[0]):
        adj2 = complete_matrix(adj2, adj1.shape[0], fill=0)
    else:
        adj1 = complete_matrix(adj1, adj2.shape[0], fill=0)

    return adj1, adj2


def calculate_edge_membership(adj):
    # 计算节点与边的对应关系，行为节点，列为边。这里为所有的边有C_n^2种排列，n为节点数量
    nodes = adj.index.tolist()
    edge_membership = [dict(zip(edge, [1, 1])) for edge in combinations(nodes, 2)]
    edge_membership = pd.DataFrame(edge_membership).fillna(0).astype(int)

    return edge_membership

# flip edges (in edge vector format)


def generate_edge_flip_vectors(edge_flips, adj, possible_edge_removes):
    n = adj.shape[0]
    adjv = np.full(int(n*(n-1)/2), 0)
    adjv[possible_edge_removes] = 1
    edge_flip_vectors = []
    for edge_filp in edge_flips.T:
        edge_flip_vector = adjv.copy()
        if not edge_filp.shape[0] == 0:
            edge_flip_vector[edge_filp] = 1 - edge_flip_vector[edge_filp]
        edge_flip_vectors.append(edge_flip_vector)

    return np.array(edge_flip_vectors)


def check_degrees_max(degree_vectors1, sorted_degrees2):
    return degree_vectors1.max(axis=1) == sorted_degrees2.max()


def check_degrees_min(degree_vectors1, sorted_degrees2):
    return degree_vectors1.min(axis=1) == sorted_degrees2.min()


def check_degrees_sorted(degree_vectors1, sorted_degrees2):
    sorted_degrees1 = np.sort(degree_vectors1, axis=1)  # (n_flip, n_nodes)
    sorted_degrees2.reshape(1, -1)  # (1, n_nodes)
    return np.all(sorted_degrees1 == sorted_degrees2, axis=1)


def flip_adj(edge_flip, adj, index2edge):
    index2edge = pd.Series(index2edge)
    # TODO: 效率比较低，待优化
    new_adj = adj.copy()
    for edge_index in edge_flip:
        edge = index2edge[edge_index]
        new_adj.loc[edge[1], edge[0]] = ~ adj.loc[edge[1], edge[0]]
    new_adj = pd.DataFrame(np.tril(new_adj.values, k=-1), index=new_adj.index, columns=new_adj.index) # 只保留下三角
    return new_adj

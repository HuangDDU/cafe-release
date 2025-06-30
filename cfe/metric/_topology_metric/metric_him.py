from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd

from ...data import simplify_networkx_network
from ...util.random_time_string import random_time_string


# 辅助函数：针对自环边插入两个新节点
def insert_two_nodes_into_selfloop(df: pd.DataFrame) -> pd.DataFrame:
    mask = df['from'] == df['to']
    df_self = df[mask]
    new_rows = []
    for _, row in df_self.iterrows():
        n = row['from']
        l = row['length']
        d = row['directed']
        newn1 = random_time_string()
        newn2 = random_time_string()
        # 将自环边拆分为两条边，每条边的长度为原始边长的 1/3
        new_rows.append({'from': n, 'to': newn1, 'length': l / 3, 'directed': d})
        new_rows.append({'from': newn1, 'to': newn2, 'length': l / 3, 'directed': d})
        new_rows.append({'from': newn2, 'to': n, 'length': l / 3, 'directed': d})
    df_non_self = df[~mask]
    df_new = pd.DataFrame(new_rows)
    return pd.concat([df_non_self, df_new], ignore_index=True)

# 辅助函数：针对重复边插入新节点
def insert_one_node_into_duplicate_edges(df: pd.DataFrame) -> pd.DataFrame:
    edge_ids = df['from'] + "#" + df['to']
    counts = edge_ids.value_counts()
    dup_edges = counts[counts >= 2].index
    mask = edge_ids.isin(dup_edges)
    new_rows = []
    for _, row in df[mask].iterrows():
        n = row['from']
        t = row['to']
        l = row['length']
        d = row['directed']
        newn = random_time_string()
        new_rows.append({'from': n, 'to': newn, 'length': l / 2, 'directed': d})
        new_rows.append({'from': newn, 'to': t, 'length': l / 2, 'directed': d})
    df_non_dup = df[~mask]
    df_new = pd.DataFrame(new_rows)
    return pd.concat([df_non_dup, df_new], ignore_index=True)

# 辅助函数：当只有一条边且非自环时，将其转换为双边(作者在这里为什么要使用abc?会对后续产生影响吗？)
def change_single_edge_into_double(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 1 and df.iloc[0]['from'] != df.iloc[0]['to']:
        row = df.iloc[0]
        new_data = [
            {'from': "a", 'to': "b", 'length': row['length'] / 2, 'directed': row['directed']},
            {'from': "b", 'to': "c", 'length': row['length'] / 2, 'directed': row['directed']}
        ]
        return pd.DataFrame(new_data)
    else:
        return df

# 根据边表构造邻接矩阵（所有出现的节点构成方阵，缺失位置填 0）
def get_adjacency_lengths(df: pd.DataFrame) -> np.ndarray:
    nodes = sorted(set(df['from']).union(set(df['to'])))
    n = len(nodes)
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    A = np.zeros((n, n))
    for _, row in df.iterrows():
        i = node2idx[row['from']]
        j = node2idx[row['to']]
        A[i, j] = row['length']
    return A

# 若矩阵维度不足，则扩展为指定大小（填充值为0）
def complete_matrix(mat: np.ndarray, size: int, fill: float = 0) -> np.ndarray:
    n, m = mat.shape
    if n == size and m == size:
        return mat
    new_mat = np.full((size, size), fill, dtype=mat.dtype)
    new_mat[:n, :m] = mat
    return new_mat

# 获取匹配的两个网络的邻接矩阵（包含网络预处理和简化流程）
def get_matched_adjacencies(net1: pd.DataFrame, net2: pd.DataFrame, simplify: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if simplify:
        directed1 = net1['directed'].any()
        directed2 = net2['directed'].any()

        def process_network(net: pd.DataFrame, directed_flag: bool) -> pd.DataFrame:
            net_proc = net.rename(columns={'length': 'weight'})
            net_proc = net_proc[~((net_proc['from'] == net_proc['to']) & (net_proc['weight'] == 0))]
            # 构造无向图
            G = nx.from_pandas_edgelist(net_proc, source='from', target='to', edge_attr='weight', create_using=nx.Graph())
            # 调用已有的简化接口
            G_simpl = simplify_networkx_network(G)
            # 将返回的 DataFrame 的列重命名为 'from', 'to', 'length'
            df_simpl = nx.to_pandas_edgelist(G_simpl)
            df_simpl = df_simpl.rename(columns={'source': 'from', 'target': 'to', 'weight': 'length'})
            df_simpl['directed'] = directed_flag

            df_simpl = insert_two_nodes_into_selfloop(df_simpl)
            df_simpl = change_single_edge_into_double(df_simpl)
            df_simpl = insert_one_node_into_duplicate_edges(df_simpl)
            return df_simpl

        net1 = process_network(net1, directed1)
        net2 = process_network(net2, directed2)

    adj1 = get_adjacency_lengths(net1)
    adj2 = get_adjacency_lengths(net2)

    size = max(adj1.shape[0], adj2.shape[0])
    if adj1.shape[0] < size:
        adj1 = complete_matrix(adj1, size, fill=0)
    if adj2.shape[0] < size:
        adj2 = complete_matrix(adj2, size, fill=0)

    return adj1, adj2

# 计算拉普拉斯矩阵 L = D - A
def laplacian_matrix(adj: np.ndarray) -> np.ndarray:
    D = np.diag(adj.sum(axis=1))
    return D - adj

# 计算 Ipsen–Mikhailov 距离（基于拉普拉斯矩阵特征值的平方根差）
def ipsen_mikhailov_distance_eigen(adj1: np.ndarray, adj2: np.ndarray) -> float:
    L1 = laplacian_matrix(adj1)
    L2 = laplacian_matrix(adj2)
    eigs1 = np.linalg.eigvalsh(L1)
    eigs2 = np.linalg.eigvalsh(L2)
    # 确保非负后取平方根
    sqrt_eigs1 = np.sqrt(np.maximum(eigs1, 0))
    sqrt_eigs2 = np.sqrt(np.maximum(eigs2, 0))
    # 两个向量间的欧氏距离
    return np.linalg.norm(sqrt_eigs1 - sqrt_eigs2)

# 计算 Hamming 边权距离（元素绝对差之和）
def hamming_distance(adj1: np.ndarray, adj2: np.ndarray) -> float:
    return np.sum(np.abs(adj1 - adj2))

# 根据公式计算 HIM 距离
def him_distance(adj1: np.ndarray, adj2: np.ndarray, gamma: float = 0.1) -> float:
    IM = ipsen_mikhailov_distance_eigen(adj1, adj2)
    H = hamming_distance(adj1, adj2)
    # 根据公式： sqrt((IM^2 + (γ*Hamming)^2)/(1+γ^2))
    return np.sqrt((IM**2 + (gamma * H)**2) / (1 + gamma**2))

# 主函数：计算 HIM 相似性度量，返回 max(0, 1 - HIM_distance)
def calculate_him(net1: pd.DataFrame, net2: pd.DataFrame, simplify: bool = False, gamma: float = 0.1) -> float:
    adj1, adj2 = get_matched_adjacencies(net1, net2, simplify=simplify)

    # 若任一邻接矩阵全为0，则返回 0
    if np.max(adj1) == 0 or np.max(adj2) == 0:
        return 0

    # 对邻接矩阵归一化，确保总和为1
    norm_adj1 = adj1 / np.sum(adj1)
    norm_adj2 = adj2 / np.sum(adj2)

    # 计算 HIM 距离，根据定义计算相似度
    distance = him_distance(norm_adj1, norm_adj2, gamma=gamma)
    similarity = max(0, 1 - distance)
    return similarity

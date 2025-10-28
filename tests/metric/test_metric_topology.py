import numpy as np
import pandas as pd
import pytest

from cfe.metric._metric_topology.metric_him import get_matched_adjacencies, him_distance
from cfe.metric.metric_topology import (
    calculate_edge_flip,
    calculate_him,
    calculate_isomorphic,
)


def test_metric_isomorphic():
    net1 = pd.DataFrame(
        data=[
            ["A", "B", 1, True],
            ["B", "C", 1, True],
            ["C", "D", 1, True],
        ],
        columns=["from", "to", "length", "direction"],
    )
    net2 = pd.DataFrame(
        data=[
            ["A", "B", 1, True],
            ["B", "C", 1, True],
            ["B", "D", 1, True],
        ],
        columns=["from", "to", "length", "direction"],
    )
    assert calculate_isomorphic(net1, net1) == 1
    assert calculate_isomorphic(net1, net2) == 0


def test_metric_flip_linear_bifurcation():
    # 对比线性拓扑和分支拓扑
    linear = pd.DataFrame(
        columns=["from", "to", "length", "directed"],
        data=[
            ["A", "B", 1, True],
            ["B", "C", 2, True],
            ["C", "D", 3, True],
        ],
    )  # 会对线性简化
    bifurcatiion = pd.DataFrame(
        data=[
            ["A", "B", 1, True],
            ["B", "C", 2, True],
            ["B", "D", 3, True],
        ],
        columns=["from", "to", "length", "directed"],
    )

    unsimplified_score = calculate_edge_flip(linear, bifurcatiion, simplify=False)
    simplified_score = calculate_edge_flip(linear, bifurcatiion, simplify=True)

    expected_unsimplified_score = 1 - 2 / 4
    expected_simplified_score = 0

    assert unsimplified_score == expected_unsimplified_score
    assert simplified_score == expected_simplified_score


def test_metric_flip_bifurcatiion_star():
    # 对比分支拓扑和星型拓扑
    bifurcatiion = pd.DataFrame(
        columns=["from", "to", "length", "directed"],
        data=[
            ["A", "B", 1, True],
            ["B", "C", 2, True],
            ["B", "D", 3, True],
            ["C", "E", 4, True],
            ["C", "F", 5, True],
        ],
    )
    star = pd.DataFrame(
        data=[
            ["A", "B", 1, True],
            ["A", "C", 2, True],
            ["A", "D", 3, True],
            ["A", "E", 4, True],
            ["A", "F", 5, True],
        ],
        columns=["from", "to", "length", "directed"],
    )
    unsimplified_score = calculate_edge_flip(bifurcatiion, star, simplify=False)
    simplified_score = calculate_edge_flip(bifurcatiion, star, simplify=True)

    expected_unsimplified_score = 1 - 4 / 8
    expected_simplified_score = expected_unsimplified_score  # 这里不会发生简化，所以分数不变

    assert unsimplified_score == expected_unsimplified_score
    assert simplified_score == expected_simplified_score


def test_metric_him_easy1():
    # 构造树形网络 net1
    # 树结构：根节点 "A"，下面三个分支 "B", "C", "D"；每个分支再扩展两个子节点
    net1_edges = [
        ("A", "B", 1.0),
        ("A", "C", 1.0),
        ("A", "D", 1.0),
        ("B", "B1", 0.8),
        ("B", "B2", 0.9),
        ("C", "C1", 0.7),
        ("C", "C2", 0.85),
        ("D", "D1", 0.95),
        ("D", "D2", 0.8),
    ]
    net1 = pd.DataFrame(net1_edges, columns=["from", "to", "length"])
    net1["directed"] = True

    # 构造相似的树形网络 net2
    # 与 net1 基本相同，仅在少数边上略有差异，例如边长度稍有调整，或者增加一条额外边
    net2_edges = [
        ("A", "B", 1.0),
        ("A", "C", 1.0),
        ("A", "D", 1.0),
        ("B", "B1", 0.8),
        ("B", "B2", 1.0),  # B2 边长度从0.9变为1.0
        ("C", "C1", 0.7),
        ("C", "C2", 0.85),
        ("D", "D1", 0.95),
        ("D", "D2", 0.8),
        # 增加一条额外的边：C->B，增加轻微的交叉关系
        ("C", "B", 0.5),
    ]
    net2 = pd.DataFrame(net2_edges, columns=["from", "to", "length"])
    net2["directed"] = True
    # 使用简化过程（如果你希望观察简化后的结果）
    sim = calculate_him(net1, net2, simplify=True, gamma=0.1)
    # 同时提取中间计算的邻接矩阵和 HIM 距离
    adj1, adj2 = get_matched_adjacencies(net1, net2, simplify=True)
    norm_adj1 = adj1 / np.sum(adj1)
    norm_adj2 = adj2 / np.sum(adj2)
    d = him_distance(norm_adj1, norm_adj2, gamma=0.1)
    # 预期相似度较高，相应 HIM 距离应较小（例如小于0.1，根据数据具体结果可能略有波动）
    assert sim > 0.9
    assert d < 0.1


def test_metric_him_easy2():
    # 构造树形网络 net3
    # 使用与上面类似的树状结构
    net3_edges = [
        ("A", "B", 1.0),
        ("A", "C", 1.0),
        ("A", "D", 1.0),
        ("B", "B1", 0.8),
        ("B", "B2", 0.9),
        ("C", "C1", 0.7),
        ("C", "C2", 0.85),
        ("D", "D1", 0.95),
        ("D", "D2", 0.8),
    ]
    net3 = pd.DataFrame(net3_edges, columns=["from", "to", "length"])
    net3["directed"] = True

    # 构造差异较大的树形网络 net4
    # 这里修改网络结构：改变分支连接和边权，令 net2 与 net1 有较大差异
    net4_edges = [
        ("A", "B", 1.0),
        ("A", "E", 1.2),  # 不再直接连接 A->C，而是 A->E
        ("B", "B1", 1.5),
        ("B", "B2", 1.4),
        ("E", "C", 0.9),
        ("E", "F", 1.1),  # E 分支出两个子节点，分别连接 C 和 F
        ("C", "C1", 0.7),
        ("C", "C2", 0.85),
        ("F", "D", 1.3),
        ("F", "D1", 1.2),  # F 分支再连接 D 和 D1，而非直接从 A->D
    ]
    net4 = pd.DataFrame(net4_edges, columns=["from", "to", "length"])
    net4["directed"] = True

    sim = calculate_him(net3, net4, simplify=False, gamma=0.1)
    adj3, adj4 = get_matched_adjacencies(net3, net4, simplify=False)
    norm_adj3 = adj3 / np.sum(adj3)
    norm_adj4 = adj4 / np.sum(adj4)
    d = him_distance(norm_adj3, norm_adj4, gamma=0.1)
    # 预期差异较大，相似度较低，HIM 距离较大（例如 HIM 距离 > 0.2，相似度 < 0.8）
    assert d > 0.2
    assert sim < 0.8


def test_metric_him_hard1():
    # 相似复杂网络测试
    # 构造 net5（含分支、环及交叉边）
    net5 = pd.DataFrame(
        {
            "from": ["A", "A", "B", "C", "B", "C", "D", "E", "F", "G", "H", "I"],
            "to": ["B", "C", "D", "D", "E", "F", "G", "F", "G", "H", "D", "H"],
            "length": [1.0, 1.2, 2.0, 2.0, 1.5, 1.5, 2.5, 1.0, 1.0, 1.8, 2.0, 2.2],
            "directed": [True] * 12,
        }
    )
    # 构造 net6，与 net5 略有不同（例如增加一条交叉边和不同边长）
    net6 = pd.DataFrame(
        {
            "from": ["A", "A", "B", "C", "B", "C", "D", "E", "F", "G", "H", "I", "E"],
            "to": ["B", "C", "D", "D", "F", "E", "G", "G", "H", "I", "D", "H", "I"],
            "length": [1.0, 1.2, 2.1, 2.0, 1.7, 1.4, 2.5, 1.1, 1.0, 1.9, 2.1, 2.2, 1.5],
            "directed": [True] * 13,
        }
    )
    # 此处采用简化过程
    sim = calculate_him(net5, net6, simplify=True, gamma=0.1)
    # 同时获取中间计算的邻接矩阵与 HIM 距离
    adj5, adj6 = get_matched_adjacencies(net5, net6, simplify=True)
    norm_adj5 = adj5 / np.sum(adj5)
    norm_adj6 = adj6 / np.sum(adj6)
    d = him_distance(norm_adj5, norm_adj6, gamma=0.1)
    # 期望在简化后两网络仍然有一定差异
    assert d > 0.05
    assert sim < 1.0


def test_metric_him_hard2():
    # （simplify=False）
    # 构造 net7（较复杂网络，保留所有原始细节）
    net7 = pd.DataFrame(
        {
            "from": ["A", "A", "B", "C", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "to": ["B", "C", "D", "D", "E", "F", "G", "F", "G", "H", "D", "H", "I"],
            "length": [1.0, 1.2, 2.0, 2.0, 1.5, 1.5, 2.5, 1.0, 1.0, 1.8, 2.0, 2.2, 1.3],
            "directed": [True] * 13,
        }
    )
    # 构造 net8，与 net7 略有不同：增加多条额外边和权重变化
    net8 = pd.DataFrame(
        {
            "from": ["A", "A", "B", "C", "B", "C", "D", "E", "F", "G", "H", "I", "J", "E", "C"],
            "to": ["B", "C", "D", "D", "F", "E", "G", "G", "H", "I", "D", "H", "I", "J", "F"],
            "length": [1.0, 1.2, 2.1, 2.0, 1.7, 1.4, 2.5, 1.1, 1.0, 1.9, 2.1, 2.2, 1.3, 1.8, 1.6],
            "directed": [True] * 15,
        }
    )
    sim = calculate_him(net7, net8, simplify=False, gamma=0.1)
    # 同时获取中间计算的邻接矩阵与 HIM 距离
    adj7, adj8 = get_matched_adjacencies(net7, net8, simplify=False)
    norm_adj7 = adj7 / np.sum(adj7)
    norm_adj8 = adj8 / np.sum(adj8)
    d = him_distance(norm_adj7, norm_adj8, gamma=0.1)
    # 在不简化的情况下，两网络的差异会更明显
    assert d > 0.05
    assert sim < 1.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])

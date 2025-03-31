import pytest
import cfe


import pandas as pd

import cfe.metric


def test_metric_isomorphic():
    net1 = pd.DataFrame(
        data=[
            ["A", "B", 1, True,],
            ["B", "C", 1, True,],
            ["C", "D", 1, True,],
        ],
        columns=["from", "to", "length", "direction"]
    )
    net2 = pd.DataFrame(
        data=[
            ["A", "B", 1, True,],
            ["B", "C", 1, True,],
            ["B", "D", 1, True,],
        ],
        columns=["from", "to", "length", "direction"]
    )
    assert cfe.metric.calc_isomorphic(net1, net1) == 1
    assert cfe.metric.calc_isomorphic(net1, net2) == 0


def test_metric_flip_linear_bifurcation():
    # 对比线性拓扑和分支拓扑
    linear = pd.DataFrame(
        columns=["from", "to", "length", "directed"],
        data=[
            ["A", "B", 1, True,],
            ["B", "C", 2, True,],
            ["C", "D", 3, True,],
        ],
    )  # 会对线性简化
    bifurcatiion = pd.DataFrame(
        data=[
            ["A", "B", 1, True,],
            ["B", "C", 2, True,],
            ["B", "D", 3, True,],
        ],
        columns=["from", "to", "length", "directed"]
    )

    unsimplified_score = cfe.metric.calc_edge_flip(linear, bifurcatiion, simplify=False)
    simplified_score = cfe.metric.calc_edge_flip(linear, bifurcatiion, simplify=True)

    expected_unsimplified_score = 1 - 2/4
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
        columns=["from", "to", "length", "directed"]
    )
    unsimplified_score = cfe.metric.calc_edge_flip(bifurcatiion, star, simplify=False)
    simplified_score = cfe.metric.calc_edge_flip(bifurcatiion, star, simplify=True)

    expected_unsimplified_score = 1 - 4/8
    expected_simplified_score = expected_unsimplified_score  # 这里不会发生简化，所以分数不变

    assert unsimplified_score == expected_unsimplified_score
    assert simplified_score == expected_simplified_score


if __name__ == "__main__":
    pytest.main(["-v", __file__])

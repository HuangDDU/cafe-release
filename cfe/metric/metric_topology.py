import networkx as nx
import pandas as pd

from ._topology_metric.metric_flip import calculate_edge_flip

# from ._topology_metric.metric_him import calculate_him


def calc_isomorphic(net1: pd.DataFrame, net2: pd.DataFrame):
    """Judge if two milestone network are  isomorphic

    Args:
        net1 (pd.DataFrame): reference milestone network
        net2 (pd.DataFrame): predict milestone network

    Returns:
        int: 0 is not isomorphic, 1 is isomorphic
    """
    # 图同构
    graph1 = nx.from_pandas_edgelist(net1, source="from", target="to")
    graph2 = nx.from_pandas_edgelist(net2, source="from", target="to")
    if nx.is_isomorphic(graph1, graph2):
        return 1
    else:
        return 0


# calc_edge_flip = calculate_edge_flip
def calc_edge_flip(
    net1: pd.DataFrame,
    net2: pd.DataFrame,
    return_type="score",
    simplify=False,  # 提前简化过了
    limit_flips=5,
    limit_combinations=12650,
):
    """Edge flip metric

    Args:
        net1 (pd.DataFrame): reference milestone network
        net2 (pd.DataFrame): predict milestone network
        return_type (str, optional): score or dict. Defaults to "score".
        simplify (bool, optional): if simplify. Defaults to False.
        limit_combinations (int, optional): filp num restriction. Defaults to 12650.

    Returns:
        _type_: _description_
    """
    calculate_edge_flip(
        net1=net1,
        net2=net2,
        return_type=return_type,
        simplify=simplify,
        limit_flips=limit_flips,
        limit_combinations=limit_combinations,
    )


# def calc_him(
#         net1: pd.DataFrame,
#         net2: pd.DataFrame,
#         simplify: bool = True,
#         gamma: float = 0.1
# ):
#     """_summary_

#     Args:
#         net1 (pd.DataFrame): _description_
#         net2 (pd.DataFrame): _description_
#         simplify (bool, optional): _description_. Defaults to True.
#         gamma (float, optional): _description_. Defaults to 0.1.
#     """
#     calculate_him(
#         net1=net1,
#         net2=net2,
#         simplify=True,
#         gamma=0.1,
#     )

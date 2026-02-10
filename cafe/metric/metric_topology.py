import networkx as nx
import pandas as pd

from . import _metric_topology


# TODO: add docs
def calculate_isomorphic(net1: pd.DataFrame, net2: pd.DataFrame):
    """Judge if two milestone network are  isomorphic

    Args:
        net1 (pd.DataFrame): reference milestone network
        net2 (pd.DataFrame): predict milestone network

    Returns:
        int: 0 is not isomorphic, 1 is isomorphic
    """
    graph1 = nx.from_pandas_edgelist(net1, source="from", target="to")
    graph2 = nx.from_pandas_edgelist(net2, source="from", target="to")
    if nx.is_isomorphic(graph1, graph2):
        return 1
    else:
        return 0


calculate_edge_flip = _metric_topology.metric_flip.calculate_edge_flip

calculate_him = _metric_topology.metric_him.calculate_him

import os.path
import pandas as pd
from .calculate_metrics import calculate_metrics
from .topology_metric import calc_isomorphic, calc_edge_flip


metrics = pd.read_csv(f"{os.path.dirname(__file__)}/metrics.csv", sep='\t')

__all__ = [
    "metrics",
    "calculate_metrics",
    "calc_isomorphic",
    "calc_edge_flip",
]

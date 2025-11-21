from .base import build_trajectory_base
from .graph_fusion import build_trajectory_graph_fusion
from .hierarchical_clustering import build_trajectory_hierarchical_clustering

LINEAGE_STRATEGIES = {
    "base": build_trajectory_base,
    "graph_fusion": build_trajectory_graph_fusion,
    "hierarchical_clustering": build_trajectory_hierarchical_clustering,
}

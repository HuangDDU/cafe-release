from . import (
    cluster,
    cycle,
    direct,
    graph,
    lineage,
    linear,
    probability,
    projection,
    velocity,
)

# The registry now maps wrapper_type to the entire module
PLOTTER_MODULE_REGISTRY = {
    "direct": direct,
    "linear": linear,
    "lineage": lineage,
    "cycle": cycle,
    "probability": probability,
    "cluster": cluster,
    "projection": projection,
    "graph": graph,
    "velocity": velocity,
}

# import scanpy as sc

from ..plot_pseudotime import plot_pseudotime_embedding, plot_pseudotime_stack

DEFAULT_MODE = "embedding"


# def plot_embedding(fadata, model_name: str = None, basis: str = None, **kwargs):
#     """Plots pseudotime on an embedding for a linear trajectory."""
#     plot_pseudotime_embedding(fadata, model_name=model_name, basis=basis, **kwargs)

plot_embedding = plot_pseudotime_embedding


# def plot_stack(fadata, model_name: str = None, cluster: str = "clusters", **kwargs):
#     """Generates a stream plot for a linear trajectory."""
#     plot_pseudotime_stack(fadata, model_name=model_name, cluster=cluster, **kwargs)


plot_stack = plot_pseudotime_stack

import scanpy as sc

from ..plot_stack import plot_stack as raw_plot_stack

DEFAULT_MODE = "embedding"


def plot_embedding(fadata, model_name: str = None, basis: str = None, **kwargs):
    """Plots pseudotime on an embedding for a linear trajectory."""
    if basis is None:
        basis = fadata.prior_information.get("basis", "X_umap")

    pseudotime = fadata.get_trajectory_pseudotime(model_name=model_name)
    fadata.obs["pseudotime"] = pseudotime

    sc.pl.embedding(fadata, color="pseudotime", basis=basis, cmap="viridis", **kwargs)


def plot_stack(fadata, model_name: str = None, cluster_key: str = "clusters", **kwargs):
    """Generates a stream plot for a linear trajectory."""
    # Directly reuse the main plot_stack function
    raw_plot_stack(fadata, cluster_key=cluster_key, model_name=model_name, **kwargs)

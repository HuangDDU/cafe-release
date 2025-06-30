from .._logging import logger
from ..data import FateAnnData
from .plot_trajectory import plot_trajectory


def plot_wrapper(
    fadata: FateAnnData,
    wrapper_type: str = None,
) -> None:
    """ plot original wrapper data

    Args:
        fadata (FateAnnData): FateAnnData object
        wrapper_type (str, optional): wrapper type determines the plot style. Defaults to None.
    """
    if wrapper_type is None:
        # extract wrapper type from fadata
        wrapper_type = fadata.wrapper_type
        logger.info(f"find wrapper type: {wrapper_type}")
    if wrapper_type == "directed":
        plot_directed(fadata)
    elif wrapper_type == "linear":
        plot_linear(fadata)
    elif wrapper_type == "cycle":
        plot_cycle(fadata)
    elif wrapper_type == "probability":
        plot_probability(fadata)
    elif wrapper_type == "cluster":
        plot_cluster(fadata)
    elif wrapper_type == "projection":
        plot_projection(fadata)
    elif wrapper_type == "graph":
        plot_graph(fadata)
    elif wrapper_type == "velocity":
        plot_velocity(fadata)


# plot_{wrapper_type}

def plot_directed(
    fadata: FateAnnData,
    color: str | list = "milestone",
):
    # # TODO: beautify
    # plot_trajectory(
    #     fadata=fadata,
    #     curve=False,
    # )
    from .plot_graph import plot_graph
    plot_graph(fadata, color=color)


def plot_linear(fadata):
    pass


def plot_cycle(fadata):
    pass


def plot_probability(fadata):
    pass


def plot_cluster(fadata):
    pass


def plot_projection(fadata):
    pass


def plot_graph(fadata):
    pass


def plot_velocity(fadata):
    import scvelo as scv

    rwd = fadata.raw_wrapper_dict

    # use velocity matrix in high dimensional space to recompute low dimensional velocity.
    # cell_index = rwd["cell_index"]
    # gene_index = rwd["gene_index"]
    # neighbors = rwd["neighbors"]
    # adata = fadata[cell_index, gene_index]
    # adata.layers["velocity"] = fadata.raw_wrapper_dict["velocity"]
    # adata.uns["neighbors"] = neighbors

    # scv.tl.velocity_graph(adata)
    # scv.pl.velocity_embedding_stream(adata, basis="umap", n_neighbors=min(neighbors["params"]["n_neighbors"], adata.shape[0]))

    # directly use velocity_adata
    velocity_adata = rwd["velocity_adata"]
    scv.pl.velocity_embedding_stream(velocity_adata)

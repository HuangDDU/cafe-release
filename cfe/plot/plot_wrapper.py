from .._logging import logger
from ..data import FateAnnData
from ..util import temporary_obsm_key

# from .plot_trajectory import plot_trajectory


def plot_wrapper(fadata: FateAnnData, wrapper_type: str = None, model_name: str = None, **kwargs) -> None:
    """plot original wrapper data

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
        plot_velocity(fadata, model_name=model_name, **kwargs)


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


def plot_linear(fadata, model_name):
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


def plot_velocity(
    fadata,
    basis=None,
    model_name: str = None,
):
    import scvelo as scv

    if basis is None:
        basis = fadata.prior_information.get("basis")
    velocity_basis = f"velocity_{basis[2:]}"
    velocity_embedding = fadata.get_raw_wrapper_dict(model_name).get(velocity_basis)

    with temporary_obsm_key(fadata, velocity_basis, velocity_embedding):
        scv.pl.velocity_embedding_stream(fadata, basis=basis[2:])

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
    # velocity_adata = rwd["velocity_adata"]

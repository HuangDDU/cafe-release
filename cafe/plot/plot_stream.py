"""
Stream Plot (Subway Map) for trajectory visualization

Visualize the distribution of cells along a trajectory in a subway-map-like format.

Inspired by phlower and STREAM
ref:
  https://phlower.readthedocs.io/en/latest/generated/phlower.ext.plot_stream.html
  https://github.com/CostaLab/phlower/blob/main/phlower/external/stream.py
  https://github.com/pinellolab/STREAM
"""

from typing import List, Tuple, Union

import matplotlib.pyplot as plt

from ..data import FateAnnData


def plot_stream(
    fadata: FateAnnData,
    model_name: str = None,
    mode: str = "cell",
    color: Union[str, List[str]] = None,
    embedding_basis: str = "X_umap",
    fig_size: Tuple[float, float] = (10, 6),
    fig_legend_ncol: int = 1,
    show_text: bool = True,
    show_graph: bool = True,
    show_legend: bool = True,
    alpha: float = 0.8,
    s: int = 30,
    save_fig: bool = False,
    fig_path: str = None,
    fig_format: str = "pdf",
    **kwargs,
) -> List[plt.Figure]:
    """
    Plot Stream graph (Subway Map).

    Visualize the distribution of cells along a trajectory in a subway-map-like format.

    Parameters
    ----------
    fadata : FateAnnData
        FateAnnData object, must have trajectory information added via add_trajectory.
    model_name : str, optional
        Name of the trajectory model to use, defaults to the current model.
    mode : str, optional
        Plotting mode, "cell" for single-cell scatter plot, "density" for density stream plot. Default "cell".
    color : str or list of str, optional
        Column name(s) for coloring (obs column or gene name), defaults to cluster column.
    embedding_basis : str, optional
        Embedding used for computing node positions. Default "X_umap".
    fig_size : tuple, optional
        Figure size. Default (10, 6).
    fig_legend_ncol : int, optional
        Number of legend columns. Default 1.
    show_text : bool, optional
        Whether to show node labels. Default True (only effective when mode="cell").
    show_graph : bool, optional
        Whether to show trajectory graph. Default True (only effective when mode="cell").
    show_legend : bool, optional
        Whether to show legend. Default True (only effective when mode="cell").
    alpha : float, optional
        Point transparency. Default 0.8 (only effective when mode="cell").
    s : int, optional
        Point size. Default 30 (only effective when mode="cell").
    save_fig : bool, optional
        Whether to save the figure. Default False.
    fig_path : str, optional
        Path to save the figure.
    fig_format : str, optional
        Format for saving. Default "pdf".
    **kwargs
        Additional arguments passed to the underlying plotting function.

    Returns
    -------
    List[plt.Figure]
        List of generated figures.

    Examples
    --------
    >>> import cafe
    >>> fadata = cafe.data.read_pancreas()
    >>> # Single-cell scatter plot mode
    >>> cafe.pl.plot_stream(fadata, mode="cell", color="clusters")
    >>> # Density stream plot mode
    >>> cafe.pl.plot_stream(fadata, mode="density", color="clusters")
    """
    from ._plot_stream.adapter import StreamPlotAdapter

    if mode not in ["cell", "density"]:
        raise ValueError(f"mode must be 'cell' or 'density', got '{mode}'")

    # Handle color parameter
    if color is None:
        color = []
    elif isinstance(color, str):
        color = [color]

    # Create adapter
    adapter = StreamPlotAdapter(fadata, model_name=model_name)

    # Prepare data
    adapter.prepare_adata_for_stream(embedding_basis=embedding_basis)

    # Call different plotting functions based on mode
    if mode == "cell":
        figs = adapter.plot_stream_sc(
            root="root",
            color=color if color else None,
            fig_size=fig_size,
            fig_legend_ncol=fig_legend_ncol,
            show_text=show_text,
            show_graph=show_graph,
            show_legend=show_legend,
            alpha=alpha,
            s=s,
            save_fig=save_fig,
            fig_path=fig_path,
            fig_format=fig_format,
            **kwargs,
        )
    else:  # mode == "density"
        figs = adapter.plot_stream(
            root="root",
            color=color if color else None,
            fig_size=fig_size,
            save_fig=save_fig,
            fig_path=fig_path,
            fig_format=fig_format,
            **kwargs,
        )

    return figs

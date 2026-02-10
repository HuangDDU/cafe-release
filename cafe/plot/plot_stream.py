"""
Stream Plot (Subway Map) for trajectory visualization

Visualize the distribution of cells along a trajectory in a subway-map-like format.

Inspired by phlower and STREAM
ref:
  https://phlower.readthedocs.io/en/latest/generated/phlower.ext.plot_stream.html
  https://github.com/CostaLab/phlower/blob/main/phlower/external/stream.py
  https://github.com/pinellolab/STREAM
"""

import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt

from .._logging import logger
from ..data import FateAnnData


# TODO: add docs
def plot_stream(
    fadata: FateAnnData,
    model_name: str = None,
    mode: str = "cell",
    color: Union[str, List[str]] = None,
    basis: str = None,
    fig_size: Tuple[float, float] = (10, 6),
    fig_legend_ncol: int = 1,
    show_text: bool = True,
    show_graph: bool = True,
    show_legend: bool = True,
    alpha: float = 0.8,
    s: int = 30,
    save: Union[bool, str] = None,
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
    basis : str, optional
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
    save : bool or str, optional
        If True, save to default path `.cafe/{fadata.id}/img/stream_{mode}_{color}.png`.
        If str, save to the specified path. Default None (don't save).
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
    >>> # Save to default path
    >>> cafe.pl.plot_stream(fadata, mode="cell", color="clusters", save=True)
    >>> # Save to custom path
    >>> cafe.pl.plot_stream(fadata, mode="cell", color="clusters", save="my_stream.png")
    """
    from ._plot_stream.adapter import StreamPlotAdapter

    if mode not in ["cell", "density"]:
        raise ValueError(f"mode must be 'cell' or 'density', got '{mode}'")

    # Handle color parameter
    if color is None:
        color = []
    elif isinstance(color, str):
        color = [color]

    if basis is None:
        basis = fadata.prior_information.get("basis")

    if "start_milestone" in fadata.prior_information:
        root_node = fadata.prior_information["start_milestone"]
    else:
        start_cell = fadata.prior_information["start_cell"]
        root_node = fadata.get_start_milestone(start_cell=start_cell)

    # Create adapter
    adapter = StreamPlotAdapter(fadata, model_name=model_name)

    # Prepare data
    adapter.prepare_adata_for_stream(embedding_basis=basis, root_node=root_node)

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
            **kwargs,
        )
    else:  # mode == "density"
        figs = adapter.plot_stream(
            root="root",
            color=color if color else None,
            fig_size=fig_size,
            **kwargs,
        )

    # Handle save parameter (plot_trajectory style)
    if save:
        if isinstance(save, bool) and save:
            # Generate default save path
            color_str = "_".join(color) if color else "default"
            model = model_name or fadata.model_name or "model"
            save_path = f".cafe/{fadata.id}/img/stream_{mode}_{model}_{color_str}.png"
        else:
            save_path = save

        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Save all figures
        for i, fig in enumerate(figs):
            if len(figs) == 1:
                actual_path = save_path
            else:
                # Multiple figures: add index to filename
                base, ext = os.path.splitext(save_path)
                actual_path = f"{base}_{i}{ext}"
            fig.savefig(actual_path, bbox_inches="tight", pad_inches=0.1)
            logger.info(f"Saved stream plot to '{actual_path}'")

    return figs

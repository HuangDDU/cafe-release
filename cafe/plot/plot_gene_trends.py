import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

from .util import save_fig


def _fit_curve_gaussian(pseudotime, expression, sigma=5.0, with_ci=True):
    """Fit smoothed curve by Gaussian filter and optional CI."""
    smooth_expr = gaussian_filter1d(expression, sigma=sigma)

    if not with_ci:
        return smooth_expr, None, None

    df = pd.DataFrame({"pt": pseudotime, "expr": expression})
    window = max(int(len(pseudotime) * 0.1), 5)
    rolling_std = df["expr"].rolling(window=window, center=True, min_periods=2).std().fillna(0).values
    smooth_std = gaussian_filter1d(rolling_std, sigma=sigma)

    window_size_actual = np.clip(np.ones_like(pseudotime) * window, 1, len(pseudotime))
    sem = smooth_std / np.sqrt(window_size_actual)
    ci = 1.96 * sem
    ci_lower = smooth_expr - ci
    ci_upper = smooth_expr + ci
    return smooth_expr, ci_lower, ci_upper


def _fit_curve_pygam(pseudotime, expression, with_ci=True):
    """Fit smoothed curve by pygam and optional CI."""
    try:
        from pygam import LinearGAM, s
    except ImportError:
        raise ImportError("Please install pygam (`pip install pygam`) to use curve_mode='pygam'")

    gam = LinearGAM(s(0)).fit(pseudotime.reshape(-1, 1), expression)
    smooth_expr = gam.predict(pseudotime.reshape(-1, 1))

    if not with_ci:
        return smooth_expr, None, None

    ci_intervals = gam.confidence_intervals(pseudotime.reshape(-1, 1), width=0.95)
    ci_lower = ci_intervals[:, 0]
    ci_upper = ci_intervals[:, 1]
    return smooth_expr, ci_lower, ci_upper


def _fit_expression_curve(pseudotime, expression, curve_mode="gaussian", sigma=5.0, with_ci=True):
    """Unified interface for expression curve fitting."""
    if curve_mode == "gaussian":
        return _fit_curve_gaussian(pseudotime, expression, sigma=sigma, with_ci=with_ci)
    if curve_mode == "pygam":
        return _fit_curve_pygam(pseudotime, expression, with_ci=with_ci)
    raise ValueError(f"Unknown curve_mode: '{curve_mode}'. Supported modes are 'gaussian' and 'pygam'.")


def plot_gene_trends(
    fadata,
    genes,
    model_name=None,
    lineages=None,
    pseudotime_key=None,
    start_milestone=None,
    show_cell=False,
    cell_color_key=None,
    figsize=None,
    return_fig=False,
    curve_mode="pygam",  # gaussian or pygam
    n_cols=4,
    sigma=5.0,
    save=None,
    **kwargs,
):
    """
    Plot gene expression trends along pseudotime for different lineages.

    Parameters
    ----------
    fadata : FateAnnData
        The FateAnnData object.
    genes : str or list of str
        The gene(s) to plot expression for.
    model_name: str, optional
        The trajectory model to use for pseudotime and lineage information. If None, uses the default model in fadata.prior_information["model_name"].
    lineages : str or list of str, optional
        The specific lineages to plot. If None, plots all available lineages.
    pseudotime_key : str, optional
        Key in `fadata.obs` containing pseudotime. If None, calculates trajectory pseudotime automatically.
    start_milestone : str, optional
        The milestone to start pseudotime computation from.
    show_cell : bool, optional
        Whether to show individual cell expression as scatter points.
    cell_color_key : str, optional
        Key in `fadata.obs` to color cells by when `show_cell=True`. If None, uses the cluster annotation specified in `fadata.prior_information["cluster"]`.
    curve_mode : str, optional
        Method to fit the gene expression curve. Options are "gaussian" for simple smoothing or "pygam" for Generalized Additive Models.
    n_cols : int, optional
        Number of columns in the subplot grid when plotting multiple genes.
    sigma : float, optional
        Smoothing parameter for Gaussian filter.
    figsize : tuple, optional
        Size of the figure.
    return_fig : bool, optional
        Whether to return the figure and axes objects.
    """
    if isinstance(genes, str):
        genes = [genes]

    if show_cell and cell_color_key is None:
        cell_color_key = fadata.prior_information["cluster"]

    # Get pseudotime
    if pseudotime_key is None:
        pseudotime = fadata.get_trajectory_pseudotime(start_milestone=start_milestone, model_name=model_name)
        if hasattr(pseudotime, "values"):
            pseudotime = pseudotime.values
        elif isinstance(pseudotime, list):
            pseudotime = np.array(pseudotime)
    else:
        pseudotime = fadata.obs[pseudotime_key].astype(float).values

    # Get lineages dict: {"Lineage_1": ["cell1", "cell2"], ...}
    lineage_dict = fadata.get_lineages(start_milestone=start_milestone, return_element_type="obs_index")
    if lineages is not None:
        if isinstance(lineages, str):
            lineages = [lineages]
        lineage_dict = {k: v for k, v in lineage_dict.items() if k in lineages}

    # Determine colors for lineages
    lineage_colors = {}
    mw = fadata.get_milestone_wrapper()
    milestone_colors = mw.milestone_color_dict
    for k in lineage_dict.keys():
        if k in milestone_colors:
            lineage_colors[k] = milestone_colors[k]

    # Fill remaining with default plotting colors
    rc = plt.rcParams["axes.prop_cycle"]
    default_colors = rc.by_key()["color"]
    for i, k in enumerate(lineage_dict.keys()):
        if k not in lineage_colors:
            lineage_colors[k] = default_colors[i % len(default_colors)]

    n_genes = len(genes)
    cols = min(n_cols, n_genes)
    rows = max(1, int(np.ceil(n_genes / cols)))

    show_scatter_legend = show_cell and (cell_color_key in fadata.obs.columns)
    show_lineage_legend = len(lineage_dict) > 0

    # Keep each axes around 5x4 inches, reserve extra right panel width for legends.
    base_axes_width = 5 * cols
    base_axes_height = 4 * rows
    if show_scatter_legend and show_lineage_legend:
        legend_panel_width = 4.8
    elif show_scatter_legend or show_lineage_legend:
        legend_panel_width = 2.8
    else:
        legend_panel_width = 0.0

    if figsize is None:
        figsize = (base_axes_width + legend_panel_width, base_axes_height)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    axes = axes.flatten()
    lineage_legend_handles = {}

    scatter_legend_handles = []
    scatter_legend_labels = []
    if show_cell and (cell_color_key in fadata.obs.columns):
        cell_series = fadata.obs[cell_color_key]
        if pd.api.types.is_categorical_dtype(cell_series):
            categories = list(cell_series.cat.categories)
        else:
            categories = sorted(cell_series.dropna().unique().tolist())

        uns_key = f"{cell_color_key}_colors"
        if (uns_key in fadata.uns) and (len(fadata.uns[uns_key]) >= len(categories)):
            cat_colors = list(fadata.uns[uns_key])[: len(categories)]
        else:
            cat_colors = [default_colors[idx % len(default_colors)] for idx in range(len(categories))]

        for category, color in zip(categories, cat_colors):
            scatter_legend_handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=color, markeredgecolor="none"))
            scatter_legend_labels.append(str(category))

    for i, gene in enumerate(genes):
        ax = axes[i]

        # Show cell, use scanpy scatter plot.
        if show_cell:
            tmp_pseudotime_key = "_tmp_pseudotime"
            fadata.obs[tmp_pseudotime_key] = pseudotime
            ax = sc.pl.scatter(
                fadata,
                x=tmp_pseudotime_key,
                y=gene,
                color=cell_color_key,
                alpha=0.5,
                ax=ax,
                size=10,
                legend_loc="none",
                show=False,
            )
            scatter_legend = ax.get_legend()
            if scatter_legend is not None:
                scatter_legend.remove()
            del fadata.obs[tmp_pseudotime_key]

        # Get gene expression
        if gene in fadata.var_names:
            expr = fadata[:, gene].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray()
            expr = expr.flatten()
        else:
            raise ValueError(f"Gene '{gene}' not found in fadata.var_names")

        for lineage_name, cell_indices in lineage_dict.items():
            if len(cell_indices) == 0:
                continue

            # Filter cells for current lineage
            mask = fadata.obs_names.isin(cell_indices)
            lineage_pseudotime = pseudotime[mask]
            lineage_expression = expr[mask]

            # Filter NaNs in pseudotime
            valid_idx = ~pd.isna(lineage_pseudotime)
            lineage_pseudotime = lineage_pseudotime[valid_idx]
            lineage_expression = lineage_expression[valid_idx]

            if len(lineage_pseudotime) == 0:
                continue

            # Sort by pseudotime
            sort_idx = np.argsort(lineage_pseudotime)
            lineage_pseudotime = lineage_pseudotime[sort_idx]
            lineage_expression = lineage_expression[sort_idx]

            smooth_expr, ci_lower, ci_upper = _fit_expression_curve(
                lineage_pseudotime,
                lineage_expression,
                curve_mode=curve_mode,
                sigma=sigma,
                with_ci=True,
            )

            color = lineage_colors[lineage_name]
            line_handle = ax.plot(lineage_pseudotime, smooth_expr, label=lineage_name, color=color, linewidth=2)[0]
            ax.fill_between(lineage_pseudotime, ci_lower, ci_upper, color=color, alpha=0.15)
            if lineage_name not in lineage_legend_handles:
                lineage_legend_handles[lineage_name] = Line2D([0], [0], color=line_handle.get_color(), lw=2)

        ax.set_title(gene)
        ax.set_xlabel("pseudotime" if pseudotime_key is None else pseudotime_key)
        ax.set_ylabel("expression")
        ax.grid(True, linestyle="-", alpha=0.5)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if show_scatter_legend or (len(lineage_legend_handles) > 0):
        # Compute the plotting area ratio from the requested figure width.
        right_limit = min(0.88, max(0.60, base_axes_width / figsize[0]))
        fig.subplots_adjust(right=right_limit, wspace=0.28, hspace=0.30)

        x_scatter = right_limit + 0.02
        x_lineage = right_limit + (0.18 if show_scatter_legend else 0.02)
    else:
        right_limit = 1.0
        x_scatter = 1.02
        x_lineage = 1.02

    # Figure-level legends on the right side in two columns:
    # left column = scatter groups, right column = lineage curves.
    if show_cell and len(scatter_legend_handles) > 0:
        legend_cells = fig.legend(
            handles=scatter_legend_handles,
            labels=scatter_legend_labels,
            title=f"{cell_color_key} (scatter)",
            loc="center left",
            bbox_to_anchor=(x_scatter, 0.5),
            frameon=False,
        )
        fig.add_artist(legend_cells)

    if len(lineage_legend_handles) > 0:
        fig.legend(
            handles=list(lineage_legend_handles.values()),
            labels=list(lineage_legend_handles.keys()),
            title="lineage (curve)",
            loc="center left",
            bbox_to_anchor=(x_lineage, 0.5),
            frameon=False,
        )

    plt.tight_layout(rect=[0, 0, right_limit, 1])

    save_fig(save, default_filename=f".cafe/{fadata.id}/img/pseudotime_embedding_({model_name}-{genes}).png", ax=ax)

    if return_fig:
        return fig, axes
    else:
        plt.show()


def plot_gene_linear_trends(
    fadata,
    genes,
    model_name=None,
    pseudotime_key=None,
    start_milestone=None,
    # show_cell=False, # TODO: show expression on plot for every gene
    # seperate_graph=False, # TODO: seprate graph and show expression for every gene
    figsize=(8, 5),
    return_fig=False,
    curve_mode="gaussian",  # gaussian or pygam
    sigma=5.0,
    linewidth=2.0,
    with_ci=True,
    ci_alpha=0.12,
    legend_loc="best",
    save=None,
    **kwargs,
):
    """Plot multiple gene trends in ONE subplot for linear trajectory.

    This function does not separate cells by lineage. It fits all selected genes
    on the same pseudotime axis and overlays their curves in one axes with
    different colors.

    Parameters
    ----------
    fadata : FateAnnData
        The FateAnnData object.
    genes : str or list of str
        One or more genes to plot in the same axes.
    model_name : str, optional
        Trajectory model name.
    pseudotime_key : str, optional
        Key in ``fadata.obs`` for pseudotime. If None, compute by trajectory.
    start_milestone : str, optional
        Start milestone used when computing pseudotime.
    show_cell: bool, optional
        Whether to show individual cell expression as scatter points.
    figsize : tuple, optional
        Figure size.
    return_fig : bool, optional
        Whether to return figure and axes.
    curve_mode : {"gaussian", "pygam"}, optional
        Curve fitting mode.
    sigma : float, optional
        Gaussian smoothing parameter.
    linewidth : float, optional
        Curve line width.
    with_ci : bool, optional
        Whether to draw confidence interval band.
    ci_alpha : float, optional
        Alpha value for confidence interval band.
    legend_loc : str, optional
        Legend location for matplotlib.
    save : str, optional
        Output figure path.
    """
    if isinstance(genes, str):
        genes = [genes]
    if len(genes) == 0:
        raise ValueError("`genes` is empty.")

    # pseudotime
    if pseudotime_key is None:
        pseudotime = fadata.get_trajectory_pseudotime(start_milestone=start_milestone, model_name=model_name)
        if hasattr(pseudotime, "values"):
            pseudotime = pseudotime.values
        elif isinstance(pseudotime, list):
            pseudotime = np.array(pseudotime)
    else:
        if pseudotime_key not in fadata.obs.columns:
            raise ValueError(f"`pseudotime_key` '{pseudotime_key}' not found in fadata.obs")
        pseudotime = fadata.obs[pseudotime_key].astype(float).values

    # keep only valid pseudotime once
    valid_pt_mask = ~pd.isna(pseudotime)
    pt = pseudotime[valid_pt_mask]
    if len(pt) == 0:
        raise ValueError("No valid pseudotime values after filtering NaNs.")

    sort_idx = np.argsort(pt)
    pt = pt[sort_idx]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    rc = plt.rcParams["axes.prop_cycle"]
    default_colors = rc.by_key()["color"]

    for i, gene in enumerate(genes):
        if gene not in fadata.var_names:
            raise ValueError(f"Gene '{gene}' not found in fadata.var_names")

        expr = fadata[:, gene].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        expr = np.asarray(expr).flatten()

        expr = expr[valid_pt_mask][sort_idx]

        smooth_expr, ci_lower, ci_upper = _fit_expression_curve(
            pt,
            expr,
            curve_mode=curve_mode,
            sigma=sigma,
            with_ci=with_ci,
        )

        color = default_colors[i % len(default_colors)]
        ax.plot(pt, smooth_expr, color=color, linewidth=linewidth, label=str(gene))
        if with_ci:
            ax.fill_between(pt, ci_lower, ci_upper, color=color, alpha=ci_alpha)

    ax.set_title("Gene linear trends")
    ax.set_xlabel("pseudotime" if pseudotime_key is None else pseudotime_key)
    ax.set_ylabel("expression")
    ax.grid(True, linestyle="-", alpha=0.5)
    ax.legend(loc=legend_loc, frameon=False)
    plt.tight_layout()

    save_fig(save, default_filename=f".cafe/{fadata.id}/img/pseudotime_linear_trends_({model_name}-{genes}).png", ax=ax)

    if return_fig:
        return fig, ax
    plt.show()

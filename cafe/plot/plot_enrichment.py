import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .util import save_fig


def plot_enrichment(
    enrich_df,
    method="auto",
    top_n=4,
    group=True,
    group_col=None,
    term_col=None,
    p_col=None,
    score_col=None,
    count_col=None,
    x_col=None,
    figsize=(8.0, 7.5),
    bar_palette="blend:#d95f5f,#4c78a8",
    cmap="coolwarm_r",
    strip_facecolor="#E5E5E5",
    edgecolor="#FFFFFF",
    show_value=True,
    title=None,
    save_path=None,
    save=False,
):
    """
    Plot pathway enrichment results from DAVID or MetaScape.

    Parameters
    ----------
    enrich_df : pd.DataFrame
        Enrichment result DataFrame.
    method : str, default='auto'
        One of {'auto', 'david', 'metascape'}.
    top_n : int, default=4
        Number of terms to show.
        If group=True, keeps top_n terms per group.
        If group=False, keeps top_n terms globally.
    group : bool, default=True
        Whether to visualize grouped facets.
        If False, plot a single panel like the example screenshot.
    group_col, term_col, p_col, score_col, count_col : str or None
        Optional manual column names. If None, infer automatically.
    x_col : str or None
        Optional x-axis column for group=False mode.
        If None, uses count_col when available, otherwise uses significance score.
    figsize : tuple, default=(8.0, 7.5)
        Figure size.
    bar_palette : str/list/dict
        Seaborn palette used in grouped mode.
    cmap : str, default='coolwarm_r'
        Colormap used in single-panel mode.
    strip_facecolor : str
        Face color of right-side facet strips in grouped mode.
    edgecolor : str
        Bar edge color.
    show_value : bool, default=True
        Whether to annotate bars with numeric values.
    title : str or None
        Plot title.
    save_path : str or None
        Output path for saving figure.
    save : bool | str | None
        Passed to save_fig. If True, save to default filename.
        If str, save to this path.
    """

    def _norm(s):
        return str(s).strip().lower().replace(" ", "").replace("_", "")

    def _pick(candidates, columns):
        norm_map = {_norm(c): c for c in columns}
        for cand in candidates:
            c = norm_map.get(_norm(cand))
            if c is not None:
                return c
        return None

    def _first_match_contains(keywords, columns):
        for c in columns:
            cn = _norm(c)
            if any(k in cn for k in keywords):
                return c
        return None

    def _to_float(series):
        return pd.to_numeric(series, errors="coerce")

    if enrich_df is None or len(enrich_df) == 0:
        raise ValueError("enrich_df is empty.")

    df = enrich_df.copy()
    columns = df.columns.tolist()

    method_l = method.lower()
    if method_l not in {"auto", "david", "metascape"}:
        raise ValueError("method must be one of ['auto', 'david', 'metascape']")

    # ------- Column inference -------
    if term_col is None:
        term_col = _pick(
            ["term", "description", "pathway", "pathway_name", "name", "term_name"],
            columns,
        )
        if term_col is None:
            term_col = _first_match_contains(["term", "desc", "pathway", "name"], columns)

    if group_col is None:
        group_col = _pick(
            ["category", "group", "source", "ontology", "collection"],
            columns,
        )
        if group_col is None:
            group_col = _first_match_contains(["category", "group", "source", "ontology", "collection"], columns)

    # Significance source priority: score_col > p_col > auto-infer.
    if score_col is None and p_col is None:
        if method_l == "david":
            p_col = _pick(["benjamini", "fdr", "pvalue", "p_value"], columns)
        elif method_l == "metascape":
            score_col = _pick(["logp", "-log10(p)", "neglog10p", "enrichment_score"], columns)
            if score_col is None:
                p_col = _pick(["pvalue", "fdr", "qvalue", "p_value"], columns)
        else:
            score_col = _pick(["logp", "-log10(p)", "neglog10p", "enrichment_score"], columns)
            if score_col is None:
                p_col = _pick(["benjamini", "fdr", "qvalue", "pvalue", "p_value"], columns)

    if count_col is None:
        count_col = _pick(["count", "gene_count", "hits", "n_genes", "list hits"], columns)

    if term_col is None:
        raise ValueError(f"Cannot infer term column from: {columns}")

    if group_col is None:
        df["__group__"] = "Enrichment"
        group_col = "__group__"

    # ------- Build significance score -------
    if score_col is not None and score_col in df.columns:
        df["__score__"] = _to_float(df[score_col])
        # Some tables store log10(p) as negative values; convert to positive significance.
        if (df["__score__"].dropna() < 0).any():
            df["__score__"] = -df["__score__"]
        score_label = score_col
        df["__pval__"] = np.power(10.0, -df["__score__"])
    elif p_col is not None and p_col in df.columns:
        pvals = _to_float(df[p_col]).clip(lower=1e-300)
        df["__score__"] = -np.log10(pvals)
        score_label = f"-log10({p_col})"
        df["__pval__"] = pvals
    else:
        auto_p = _pick(["benjamini", "fdr", "qvalue", "pvalue", "p_value"], columns)
        if auto_p is None:
            raise ValueError("Cannot infer significance column. Please set score_col (already -log10 transformed) or p_col.")
        pvals = _to_float(df[auto_p]).clip(lower=1e-300)
        df["__score__"] = -np.log10(pvals)
        score_label = f"-log10({auto_p})"
        df["__pval__"] = pvals

    # ------- Normalize term/group text -------
    df["__term__"] = df[term_col].astype(str)
    # DAVID common format: 'GO:xxxx~description' -> keep description only.
    df["__term__"] = df["__term__"].str.replace(r"^[^~]+~", "", regex=True)
    df["__group__"] = df[group_col].astype(str)

    if count_col is not None and count_col in df.columns:
        df["__count__"] = _to_float(df[count_col])
    else:
        df["__count__"] = np.nan

    plot_df = df[["__group__", "__term__", "__score__", "__count__", "__pval__"]].dropna(subset=["__score__"])
    if plot_df.empty:
        raise ValueError("No valid rows after parsing score column.")

    # Keep top terms.
    if group:
        plot_df = plot_df.sort_values(["__group__", "__score__"], ascending=[True, False]).groupby("__group__", group_keys=False).head(top_n).copy()
    else:
        plot_df = plot_df.sort_values("__score__", ascending=False).head(top_n).copy()

    if x_col is not None and x_col in df.columns:
        plot_df["__x__"] = _to_float(df.loc[plot_df.index, x_col])
        x_label = x_col
    elif (not group) and (count_col is not None) and (count_col in df.columns):
        plot_df["__x__"] = plot_df["__count__"]
        x_label = count_col
    else:
        plot_df["__x__"] = plot_df["__score__"]
        x_label = score_label

    # In single-panel mode, fallback to score if chosen x-axis is non-numeric.
    if (not group) and plot_df["__x__"].dropna().empty:
        plot_df["__x__"] = plot_df["__score__"]
        x_label = score_label

    plot_df = plot_df.dropna(subset=["__x__"])
    if plot_df.empty:
        raise ValueError("No valid rows after parsing x-axis values.")

    if save_path is not None:
        save = save_path

    sns.set_theme(style="whitegrid", font_scale=1.1)

    if group:
        # ------- Grouped faceted mode -------
        group_order = plot_df.groupby("__group__")["__score__"].max().sort_values(ascending=False).index.tolist()

        g = sns.FacetGrid(
            plot_df,
            row="__group__",
            row_order=group_order,
            sharex=True,
            sharey=False,
            height=max(1.2, figsize[1] / max(len(group_order), 1)),
            aspect=figsize[0] / max(figsize[1], 0.1),
            despine=False,
        )

        def _draw(data, **kwargs):
            data = data.sort_values("__x__", ascending=True)
            ax = plt.gca()
            sns.barplot(
                data=data,
                y="__term__",
                x="__x__",
                orient="h",
                palette=bar_palette,
                edgecolor=edgecolor,
                linewidth=0.8,
                ax=ax,
            )

            if show_value:
                x_max = data["__x__"].max() if len(data) else 0
                for i, (_, row) in enumerate(data.iterrows()):
                    ax.text(
                        row["__x__"] + max(0.02 * x_max, 0.03),
                        i,
                        f"{row['__x__']:.2f}",
                        va="center",
                        ha="left",
                        fontsize=9,
                        color="#444444",
                    )

            ax.grid(axis="x", linestyle="-", alpha=0.22)
            ax.grid(axis="y", visible=False)
            ax.set_ylabel("")
            ax.set_xlabel("")

        g.map_dataframe(_draw)

        for ax, group_name in zip(g.axes.flat, group_order):
            ax.set_title("")
            ax.text(
                1.01,
                0.5,
                group_name,
                transform=ax.transAxes,
                rotation=-90,
                va="center",
                ha="left",
                fontsize=10,
                color="#666666",
                bbox=dict(boxstyle="square,pad=0.20", facecolor=strip_facecolor, edgecolor="#BDBDBD"),
            )

        g.fig.subplots_adjust(hspace=0.12, right=0.86)
        g.set_xlabels(x_label)

        if title is None:
            title = f"Pathway Enrichment ({method_l})"
        g.fig.suptitle(title, y=1.01, fontsize=14, weight="bold")

        save_fig(save, default_filename="plot_enrichment.png", ax=g.fig)
        plt.show()
        return g

    # ------- Single panel mode -------
    plot_df = plot_df.sort_values("__x__", ascending=True)
    fig, ax = plt.subplots(figsize=figsize)

    pvals = plot_df["__pval__"].clip(lower=1e-300)
    vmin = float(pvals.min())
    vmax = float(pvals.max())
    if vmax <= vmin:
        vmax = vmin + 1e-12

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    bar_colors = [cmap_obj(norm(v)) for v in pvals]

    ax.barh(
        y=plot_df["__term__"],
        width=plot_df["__x__"],
        color=bar_colors,
        edgecolor=edgecolor,
        linewidth=0.8,
    )

    if show_value:
        x_max = plot_df["__x__"].max() if len(plot_df) else 0
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(
                row["__x__"] + max(0.02 * x_max, 0.03),
                i,
                f"{row['__x__']:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#444444",
            )

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, pad=0.02)
    if p_col is not None:
        cbar.set_label(p_col)
    else:
        cbar.set_label("p-value")

    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="-", alpha=0.22)
    ax.grid(axis="y", alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is None:
        title = f"Pathway Enrichment ({method_l}, single panel)"
    ax.set_title(title, fontsize=14, weight="bold")

    plt.tight_layout()
    save_fig(save, default_filename="plot_enrichment.png", ax=fig)
    plt.show()
    return fig

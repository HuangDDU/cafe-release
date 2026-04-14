import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .util import save_fig


def plot_volcano(
    de_df,
    title="Volcano Plot",
    genes_to_highlight=None,
    lfc_thresh=1.0,
    use_padjust=True,
    significance_col=None,
    significance_threshold=0.05,
    max_log_q=15,
    point_size=18,
    point_alpha=0.65,
    palette=None,
    highlight_color="black",
    highlight_size=50,
    highlight_edgecolor="white",
    highlight_edgewidth=0.8,
    label_fontsize=10,
    label_weight="bold",
    label_dx=0.25,
    label_dy=0.25,
    show_arrow=True,
    arrow_color="black",
    arrow_style="->",
    arrow_linewidth=1.2,
    arrow_alpha=1.0,
    show=True,
    save=True,
):
    # adapt to different column names in scanpy DE results
    def _pick_first_existing(candidates, columns):
        for c in candidates:
            if c in columns:
                return c
        return None

    gene_col = _pick_first_existing(["names", "gene", "gene_name"], de_df.columns)
    lfc_col = _pick_first_existing(["logfoldchanges", "log2fc", "log2_fold_change", "log2.foldchange."], de_df.columns)
    if significance_col is not None:
        if significance_col not in de_df.columns:
            raise ValueError(f"significance_col '{significance_col}' is not in de_df columns: {list(de_df.columns)}")
        sig_col = significance_col
        sig_label = significance_col
    else:
        if use_padjust:
            sig_col = _pick_first_existing(["pvals_adj", "qvalue", "qval", "padj"], de_df.columns)
            sig_label = "adjusted p-value"
        else:
            sig_col = _pick_first_existing(["pvals", "pvalue", "p_val", "p"], de_df.columns)
            sig_label = "p-value"

    if gene_col is None or lfc_col is None or sig_col is None:
        raise ValueError(
            f"can't recognize key columns. Current columns are: {list(de_df.columns)}\n"
            "need gene column (names/gene/gene_name), logFC column (logfoldchanges/log2fc/log2_fold_change/log2.foldchange.), and significance column "
            "(adjusted: pvals_adj/qvalue/qval/padj or raw: pvals/pvalue/p_val/p)"
        )
    plot_df = de_df[[gene_col, lfc_col, sig_col]].copy()
    plot_df.columns = ["gene", "log2fc", "sig_p"]
    plot_df = plot_df.dropna(subset=["gene", "log2fc", "sig_p"])
    plot_df = plot_df.sort_values("sig_p").drop_duplicates("gene", keep="first")
    plot_df["neglog10_sig"] = -np.log10(plot_df["sig_p"].replace(0, 1e-300))

    if max_log_q is not None:
        plot_df["plot_y"] = plot_df["neglog10_sig"].clip(upper=max_log_q)
    else:
        plot_df["plot_y"] = plot_df["neglog10_sig"]

    # filter de_df and set UP/DOWN/NS
    plot_df["sig"] = "NS"
    plot_df.loc[(plot_df["sig_p"] < significance_threshold) & (plot_df["log2fc"] > lfc_thresh), "sig"] = "Up"
    plot_df.loc[(plot_df["sig_p"] < significance_threshold) & (plot_df["log2fc"] < -lfc_thresh), "sig"] = "Down"

    # core
    if genes_to_highlight is None:
        genes_to_highlight = []

    if palette is None:
        palette = {"Up": "#d62728", "Down": "#1f77b4", "NS": "#c7c7c7"}

    plt.figure(figsize=(8, 6), dpi=140)
    ax = sns.scatterplot(
        data=plot_df,
        x="log2fc",
        y="plot_y",
        hue="sig",
        palette=palette,
        s=point_size,
        alpha=point_alpha,
        linewidth=0,
    )

    # threshold lines
    ax.axvline(lfc_thresh, color="grey", linestyle="--", linewidth=1)
    ax.axvline(-lfc_thresh, color="grey", linestyle="--", linewidth=1)
    ax.axhline(-np.log10(significance_threshold), color="grey", linestyle="--", linewidth=1)

    # label genes
    for g in genes_to_highlight:
        gene_rows = plot_df.loc[plot_df["gene"] == g]
        if gene_rows.empty:
            continue

        row = gene_rows.iloc[0]
        x, y = row["log2fc"], row["plot_y"]
        ax.scatter([x], [y], s=highlight_size, c=highlight_color, edgecolors=highlight_edgecolor, linewidths=highlight_edgewidth, zorder=5)
        ax.annotate(
            g,
            xy=(x, y),
            xytext=(x + label_dx, y + label_dy),
            fontsize=label_fontsize,
            weight=label_weight,
            color=arrow_color,
            arrowprops=dict(arrowstyle=arrow_style, color=arrow_color, lw=arrow_linewidth, alpha=arrow_alpha) if show_arrow else None,
        )

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("log2 Fold Change")
    if max_log_q is None:
        ax.set_ylabel(f"-log10({sig_label})")
    else:
        ax.set_ylabel(f"-log10({sig_label}) [Capped at {max_log_q}]")
        ax.set_ylim(-0.2, max_log_q + (max_log_q * 0.05))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    save_fig(save, default_filename="plot_volcano.png", ax=ax)
    if show:
        plt.show()

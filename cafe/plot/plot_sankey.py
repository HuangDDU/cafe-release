"""Sankey visualization of raw time-pair transport support."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path


def plot_sankey(
    adata,
    tmaps: dict,
    time_key: str = "time",
    cluster_key: str = "clusters",
    palette: dict = None,
    ax=None,
    figsize: tuple = (14, 5),
    alpha: float = 0.35,
    title: str = "Cell-state flow from temporal matching",
    save: str = None,
):
    """Plot time-state nodes and ribbons whose widths are raw tmap support.

    Matrices follow obs order within each time. No trajectory is saved or input
    mutated. Node height is max(incoming, outgoing) support, not necessarily the
    observed cell count. For unit-weight complete simulation matching they agree;
    OT mass or incomplete matching must not be read as absolute cell abundance.
    All positive transitions are shown, independent of trajectory edge pruning.
    """
    from ..data._time_trajectory.timezone import _aggregate_time_maps

    # ========== Step 1: Aggregate raw transport mass by time and cell type ==========
    tables, _, _, _ = _aggregate_time_maps(adata, tmaps, time_key, cluster_key)
    nodes = tables["temporal_nodes"].rename(columns={"node_id": "milestone_id", "group": "state"})
    # Ribbon widths use raw OT mass, not row-normalized trajectory support.
    edges = (
        tables["temporal_edges"]
        .drop(columns="support")
        .rename(columns={"source": "from", "target": "to", "source_time": "time_start", "target_time": "time_end", "raw_support": "support"})
    )
    if edges.empty:
        raise ValueError("No positive transport support to draw in the Sankey diagram.")
    # ========== Step 2: Resolve colors and a consistent state order across times ==========
    states = list(dict.fromkeys(nodes["state"]))
    if palette is None:
        color_key = f"{cluster_key}_colors"
        if color_key in adata.uns:
            palette = dict(zip(adata.obs[cluster_key].cat.categories.astype(str), adata.uns[color_key]))
        else:
            cmap = plt.get_cmap("tab10")
            palette = {state: cmap(index % 10) for index, state in enumerate(states)}
    order = {state: rank for rank, state in enumerate(palette)}
    nodes = nodes.assign(order=nodes["state"].map(order)).sort_values(["time", "order"])
    # ========== Step 3: Size and position nodes within each time layer ==========
    inflow = edges.groupby("to")["support"].sum()
    outflow = edges.groupby("from")["support"].sum()
    # Use the larger flow so both incoming and outgoing ribbons fit each node.
    heights = {mid: max(float(inflow.get(mid, 0)), float(outflow.get(mid, 0))) for mid in nodes["milestone_id"]}
    times = sorted(nodes["time"].unique())
    scale = max(sum(heights[mid] for mid in layer["milestone_id"]) for _, layer in nodes.groupby("time"))
    gap = scale * 0.035
    total_height = max(sum(heights[mid] for mid in layer["milestone_id"]) + gap * (len(layer) - 1) for _, layer in nodes.groupby("time"))
    width = min(np.diff(times)) * 0.105
    location = {}
    for time, layer in nodes.groupby("time", sort=True):
        # Center layers vertically while keeping a shared mass scale.
        y = (total_height - sum(heights[mid] for mid in layer["milestone_id"]) - gap * (len(layer) - 1)) / 2
        for row in layer.itertuples():
            location[row.milestone_id] = (time, y, row.state)
            y += heights[row.milestone_id] + gap
    # ========== Step 4: Draw source-colored ribbons between adjacent time layers ==========
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    # Stack incoming and outgoing ribbons independently within each node.
    source_offset = {mid: y for mid, (_, y, _) in location.items()}
    target_offset = source_offset.copy()
    rank = {mid: index for index, mid in enumerate(nodes["milestone_id"])}
    links = sorted(edges.to_dict("records"), key=lambda edge: (edge["time_start"], rank[edge["from"]], rank[edge["to"]]))
    for edge in links:
        source, target, value = edge["from"], edge["to"], edge["support"]
        x0, _, state = location[source]
        x1, _, _ = location[target]
        x0, x1 = x0 + width / 2, x1 - width / 2
        y0, y1 = source_offset[source], target_offset[target]
        middle = (x0 + x1) / 2
        # Two cubic Bezier curves form a closed ribbon with thickness equal to transport mass.
        vertices = [
            (x0, y0),
            (middle, y0),
            (middle, y1),
            (x1, y1),
            (x1, y1 + value),
            (middle, y1 + value),
            (middle, y0 + value),
            (x0, y0 + value),
            (x0, y0),
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(vertices, codes), facecolor=palette[state], edgecolor="none", alpha=alpha, zorder=1))
        source_offset[source] += value
        target_offset[target] += value
    # ========== Step 5: Overlay node bars, state labels, and observation times ==========
    for mid, (time, y, state) in location.items():
        height = heights[mid]
        ax.add_patch(Rectangle((time - width / 2, y), width, height, facecolor=palette[state], edgecolor="white", linewidth=1.2, zorder=2))
        if height > 0:
            ax.text(time, y + height / 2, state, ha="center", va="center", color="white", fontweight="bold", zorder=3)
    for time in times:
        ax.text(time, total_height + scale * 0.055, f"t = {time:g}", ha="center", va="bottom")
    # ========== Step 6: Finalize the layout and optionally save the figure ==========
    ax.set_xlim(times[0] - width * 1.8, times[-1] + width * 1.8)
    ax.set_ylim(-scale * 0.025, total_height + scale * 0.14)
    ax.set_title(title, pad=12)
    ax.axis("off")
    ax.figure.tight_layout()
    if save:
        ax.figure.savefig(save, dpi=180, bbox_inches="tight")
    return ax

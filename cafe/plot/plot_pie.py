import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .. import logger
from ..data import FateAnnData


# TODO: add docs
def plot_pie(
    fadata: FateAnnData,
    model_name: str = None,
    color: str = None,
    ax=None,
    figsize=(8, 8),
    node_size_scale: float = 1.0,
    edge_width: float = 1.0,
    show_labels: bool = True,
    layout: str = "dot",
    **kwargs,
):
    """
    Plot the milestone network with pie charts on nodes representing cell composition.

    Args:
        fadata: FateAnnData object.
        model_name: Name of the model to plot.
        color: Column in fadata.obs to use for pie chart sectors (e.g., 'cell_type').
        ax: Matplotlib axis.
        figsize: Figure size if ax is not provided.
        node_size_scale: Scaling factor for node sizes.
        edge_width: Width of edges.
        show_labels: Whether to show node labels.
        layout: Graphviz layout program (e.g., 'dot', 'neato').
    """
    if model_name is None:
        model_name = fadata.model_name
    if color is None:
        color = fadata.prior_information.get("cluster")

    # 1. Get Milestone Network and Wrapper
    mw = fadata.get_milestone_wrapper(model_name=model_name)
    milestone_network = mw.milestone_network
    is_directed = mw.directed

    # if not mw.wrapper_type == "cluster":
    if not mw.milestone_percentages["percentage"].isin([0, 1]).all():  # for old MilestoneWrapper without wrapper_type attribute
        # for wrappers except cluster wrapper, should first group onto the nearest milestones
        logger.warning("the wrapper type is not cluster, try to group cells onto the nearest milestone")
        mw = mw.group_onto_nearest_milestones()

    # 2. Build Graph
    G = nx.from_pandas_edgelist(
        milestone_network,
        source="from",
        target="to",
        create_using=nx.DiGraph if is_directed else nx.Graph,
    )

    # Ensure all milestones are in the graph
    all_milestones = set(mw.id_list)
    for m in all_milestones:
        if m not in G:
            G.add_node(m)

    # 3. Compute Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog=layout)
    except ImportError:
        logger.warning("pygraphviz not found, using spring layout")
        pos = nx.spring_layout(G)
    except Exception as e:
        logger.warning(f"Graphviz layout failed: {e}, using spring layout")
        pos = nx.spring_layout(G)

    # 4. Prepare Plot
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # 5. Draw Edges
    # Draw edges without arrows first
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_width,
        node_size=0,
        edge_color="gray",
        alpha=0.6,
        arrows=False,
        # connectionstyle="arc3,rad=0.1",
    )

    # Draw arrows separately, placing them at the midpoint of each edge to avoid being covered by large pie charts
    for from_id, to_id in G.edges():
        x0, y0 = pos[from_id]
        x1, y1 = pos[to_id]
        # xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        # Shorten arrow length to avoid pointing to the center of the pie chart
        shrink = 0.18  # Adjustable, larger value means further away from node
        # xm = x0 + dx * (0.5 - shrink / 2)
        # ym = y0 + dy * (0.5 - shrink / 2)
        ax.annotate(
            "",
            xy=(x1 - dx * shrink, y1 - dy * shrink),
            xytext=(x0 + dx * shrink, y0 + dy * shrink),
            arrowprops=dict(
                arrowstyle="-|>",
                color="gray",
                lw=edge_width,
                alpha=0.7,
                shrinkA=0,
                shrinkB=0,
            ),
            annotation_clip=False,
        )

    # 6. Prepare Colors
    if color in fadata.obs:
        # Try to get colors from uns
        if f"{color}_colors" in fadata.uns:
            try:
                categories = fadata.obs[color].cat.categories
                colors = fadata.uns[f"{color}_colors"]
                color_map = dict(zip(categories, colors))
            except Exception:
                import seaborn as sns

                unique_vals = fadata.obs[color].unique()
                palette = sns.color_palette("tab20", len(unique_vals))
                color_map = dict(zip(unique_vals, palette))
        else:
            import seaborn as sns

            unique_vals = fadata.obs[color].unique()
            palette = sns.color_palette("tab20", len(unique_vals))
            color_map = dict(zip(unique_vals, palette))
    else:
        logger.warning(f"Column '{color}' not found in fadata.obs")
        return ax

    # 7. Draw Nodes (Pie Charts)
    mp = mw.milestone_percentages
    # Group by milestone to get cells
    # Assuming mp has columns: cell_id, milestone_id, percentage
    # We assign cells to their max percentage milestone for the pie chart
    # Or if it's a cluster wrapper, it's already 1-to-1.

    # Efficiently find cells for each milestone
    # First, get the dominant milestone for each cell
    if "percentage" in mp.columns:
        # Sort by percentage and drop duplicates to keep the highest
        dominant_milestones = mp.sort_values("percentage", ascending=False).drop_duplicates("cell_id")
    else:
        dominant_milestones = mp

    # Group cells by milestone
    cells_by_milestone = dominant_milestones.groupby("milestone_id")["cell_id"].apply(list)

    # Calculate max node size for scaling
    max_cells = 0
    for node in G.nodes():
        if node in cells_by_milestone:
            max_cells = max(max_cells, len(cells_by_milestone[node]))

    # Scale factor to make the largest node reasonable size in the plot
    # Get plot limits to estimate scale
    # But limits are not set yet.
    # Let's use a heuristic based on coordinate range.
    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]
    if x_values and y_values:
        x_range = max(x_values) - min(x_values)
        y_range = max(y_values) - min(y_values)
        plot_span = max(x_range, y_range)
        if plot_span == 0:
            plot_span = 1
    else:
        plot_span = 1

    base_radius = plot_span * 0.02 * node_size_scale

    # Ensure the main plot is a perfect circle
    ax.set_aspect("equal")

    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]

        if node not in cells_by_milestone:
            # Draw empty node
            ax.scatter(x, y, s=50, c="lightgray", edgecolors="gray")
            if show_labels:
                ax.text(x, y, str(node), ha="center", va="center", fontsize=8)
            continue

        cells = cells_by_milestone[node]
        cell_data = fadata.obs.loc[cells, color]
        counts = cell_data.value_counts()

        if counts.sum() == 0:
            continue

        # Radius proportional to sqrt of cell count
        radius = base_radius * np.sqrt(len(cells) / max_cells)
        # Ensure a minimum size
        radius = max(radius, base_radius * 0.3)

        # Use inset_axes to ensure the pie chart is a perfect circle and centered
        pie_size = radius * 2
        inset_ax = ax.inset_axes([x - pie_size / 2, y - pie_size / 2, pie_size, pie_size], transform=ax.transData)
        # Pie chart data
        sizes = counts.sort_index().values
        pie_colors = [color_map.get(label, "gray") for label in counts.sort_index().index]
        inset_ax.pie(sizes, labels=None, colors=pie_colors)
        inset_ax.set_aspect("equal")
        inset_ax.axis("off")

        if show_labels:
            # Show node name and cell count
            label_text = f"{node}({len(cells)})"
            ax.text(x, y + radius * 1.15, label_text, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axis("off")
    ax.set_title(f"{model_name} ({color})")
    return ax

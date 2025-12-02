import networkx as nx
import pandas as pd

from ..._logging import logger


def build_trajectory_graph_fusion(fadata, probability: pd.DataFrame, cluster_key: str = None, new_cluster_list: list = None):
    """
    Builds a trajectory using the graph-based path-merging and pruning algorithm.
    This function is intended to be called by FateAnnData.add_trajectory_lineage.
    """
    from ...util import project_to_segments

    # --- 1. Input Preparation (largely the same as before) ---
    if cluster_key:
        cluster_series = fadata.obs[cluster_key]
    else:
        cluster_series = pd.Series(new_cluster_list, index=fadata.obs.index)

    terminal_states = probability.columns.tolist()
    prob_with_clusters = probability.join(cluster_series.rename("cluster"))
    cluster_probability = prob_with_clusters.groupby("cluster").mean()

    # --- 2. Identify Root and Construct "Pure" Individual Paths ---
    root_milestone = cluster_probability.sum(axis=1).idxmin()
    logger.debug(f"Strategy 'graph_fusion': Identified root milestone: '{root_milestone}'", indent_level=3)

    all_paths = {}
    for state in terminal_states:
        other_terminal_states = [s for s in terminal_states if s != state]
        candidate_clusters = [c for c in cluster_probability.index if c not in other_terminal_states]

        lineage_prob_series = cluster_probability.loc[candidate_clusters, state]
        sorted_clusters = lineage_prob_series.sort_values().index.tolist()

        try:
            root_idx = sorted_clusters.index(root_milestone)
            state_idx = sorted_clusters.index(state)
            if root_idx < state_idx:
                path = sorted_clusters[root_idx : state_idx + 1]
                all_paths[state] = path
        except ValueError:
            continue

    if not all_paths:
        raise ValueError("Strategy 'graph_fusion': Could not construct any valid lineage paths.")

    # --- 3. Fuse Paths into a Weighted Consensus Graph ---
    consensus_graph = nx.DiGraph()
    for path in all_paths.values():
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if consensus_graph.has_edge(u, v):
                consensus_graph[u][v]["weight"] += 1
            else:
                consensus_graph.add_edge(u, v, weight=1)

    # --- 4. Prune Graph to get Final Milestone Network ---
    final_graph = nx.DiGraph()
    nodes_to_visit = [root_milestone]
    visited_nodes = {root_milestone}
    while nodes_to_visit:
        current_node = nodes_to_visit.pop(0)
        successors = list(consensus_graph.successors(current_node))
        if not successors:
            continue

        for succ in successors:
            if succ in terminal_states:
                final_graph.add_edge(current_node, succ)
                visited_nodes.add(succ)

        non_terminal_successors = {s: consensus_graph[current_node][s]["weight"] for s in successors if s not in terminal_states}
        if non_terminal_successors:
            best_successor = max(non_terminal_successors, key=non_terminal_successors.get)
            final_graph.add_edge(current_node, best_successor)
            if best_successor not in visited_nodes:
                nodes_to_visit.append(best_successor)
                visited_nodes.add(best_successor)

    milestone_network = nx.to_pandas_edgelist(final_graph, source="from", target="to")
    milestone_network["length"] = 1.0
    milestone_network["directed"] = True

    # --- 5. Generate Progressions and Divergence Regions ---
    # (This part is simplified for brevity, assuming it's similar to your existing code)
    proj = project_to_segments(
        x=probability,
        segment_start=cluster_probability.loc[milestone_network["from"]],
        segment_end=cluster_probability.loc[milestone_network["to"]],
    )
    progressions = milestone_network.iloc[proj["segment"] - 1][["from", "to"]].copy()
    progressions["cell_id"] = fadata.obs.index
    progressions["percentage"] = proj["progression"]
    progressions = progressions[["cell_id", "from", "to", "percentage"]].reset_index(drop=True)

    divergence_nodes = [node for node, degree in final_graph.out_degree() if degree > 1]
    divergence_regions_list = []
    for i, div_node in enumerate(divergence_nodes):
        region_milestones = [div_node] + list(final_graph.successors(div_node))
        divergence_regions_list.append(
            pd.DataFrame(
                {
                    "milestone_id": region_milestones,
                    "divergence_id": f"Div_{i+1}",
                    "is_start": [True] + [False] * len(list(final_graph.successors(div_node))),
                }
            )
        )
    divergence_regions = pd.concat(divergence_regions_list, ignore_index=True) if divergence_regions_list else None

    return {
        "milestone_network": milestone_network,
        "divergence_regions": divergence_regions,
        "progressions": progressions,
    }

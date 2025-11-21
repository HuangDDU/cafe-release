import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist, pdist

from ..._logging import logger


def build_trajectory_hierarchical_clustering(
    fadata,
    probability: pd.DataFrame,
    cluster_key: str = None,
    new_cluster_list: list = None,
    distance_metric: str = "correlation",
    linkage_method: str = "average",
    map_to_clusters: bool = True,
):
    """
    Builds a trajectory using hierarchical clustering of terminal states.

    If `map_to_clusters` is True, it maps the resulting abstract branch points
    to the best-matching real cell clusters.
    """
    from ...util import project_to_segments

    # --- 1. Input Preparation ---
    if cluster_key:
        cluster_series = fadata.obs[cluster_key]
    else:
        cluster_series = pd.Series(new_cluster_list, index=fadata.obs.index)

    terminal_states = probability.columns.tolist()
    cluster_probability = probability.groupby(cluster_series).mean()

    # --- 2. Prepare Features for Terminal States ---
    terminal_features_list = []
    for state in terminal_states:
        if state in cluster_probability.index:
            terminal_features_list.append(cluster_probability.loc[state])
        else:
            logger.warning(f"Terminal state '{state}' is not a cluster name. Creating a virtual one-hot feature vector for it.")
            one_hot_vec = pd.Series(np.zeros(len(terminal_states)), index=terminal_states, name=state)
            one_hot_vec[state] = 1.0
            terminal_features_list.append(one_hot_vec)
    terminal_features = pd.concat(terminal_features_list, axis=1).T

    # --- 3. Hierarchical Clustering ---
    linkage_matrix = linkage(pdist(terminal_features, metric=distance_metric), method=linkage_method)

    # --- 4. Iteratively Build Tree and Optionally Match Branch Points ---
    milestone_network_data = []
    # This dictionary stores node info: key=node_id, value={"name": cluster_name, "feature": vector}
    node_info = {i: {"name": name, "feature": terminal_features.loc[name].values} for i, name in enumerate(terminal_states)}

    if map_to_clusters:
        global_root_milestone = cluster_probability.sum(axis=1).idxmin()
        candidate_branch_clusters = [c for c in cluster_probability.index if c not in terminal_states and c != global_root_milestone]
        used_branch_clusters = set()
        logger.debug(f"Strategy 'hierarchical': Mapping to real clusters. Root='{global_root_milestone}'.", indent_level=3)
    else:
        global_root_milestone = "Root"  # Use a generic root name if not mapping
        logger.debug("Strategy 'hierarchical': Using abstract branch names.", indent_level=3)

    for i, merge in enumerate(linkage_matrix):
        child1_idx, child2_idx = int(merge[0]), int(merge[1])
        child1_feature = node_info[child1_idx]["feature"]
        child2_feature = node_info[child2_idx]["feature"]
        parent_feature = (child1_feature + child2_feature) / 2.0

        parent_name = f"Branch_{i+1}"  # Default abstract name
        if map_to_clusters:
            available_candidates = [c for c in candidate_branch_clusters if c not in used_branch_clusters]
            if available_candidates:
                candidate_features = cluster_probability.loc[available_candidates].values
                distances = cdist(parent_feature.reshape(1, -1), candidate_features, metric=distance_metric)[0]
                best_match_idx = np.argmin(distances)
                parent_name = available_candidates[best_match_idx]
                used_branch_clusters.add(parent_name)
                logger.debug(f"Matched new branch point to real cluster: '{parent_name}'", indent_level=4)
            else:
                logger.warning(f"No available candidate clusters left. Using abstract name '{parent_name}'.")

        new_node_id = len(terminal_states) + i
        node_info[new_node_id] = {"name": parent_name, "feature": parent_feature}

        milestone_network_data.append({"from": parent_name, "to": node_info[child1_idx]["name"]})
        milestone_network_data.append({"from": parent_name, "to": node_info[child2_idx]["name"]})

    last_branch_name = node_info[len(node_info) - 1]["name"]
    milestone_network_data.append({"from": global_root_milestone, "to": last_branch_name})

    milestone_network = pd.DataFrame(milestone_network_data)
    milestone_network["length"] = 1.0
    milestone_network["directed"] = True

    # --- 5. Generate Progressions and Divergence Regions ---
    # Create a feature map for all milestones in the final network for projection
    milestone_features = {info["name"]: info["feature"] for info in node_info.values()}
    if map_to_clusters:
        milestone_features[global_root_milestone] = cluster_probability.loc[global_root_milestone].values
    else:  # For abstract "Root", use the feature of the real root cluster
        real_root_name = cluster_probability.sum(axis=1).idxmin()
        milestone_features[global_root_milestone] = cluster_probability.loc[real_root_name].values

    milestone_features_df = pd.DataFrame.from_dict(milestone_features, orient="index", columns=terminal_states)

    proj = project_to_segments(
        x=probability,
        segment_start=milestone_features_df.loc[milestone_network["from"]],
        segment_end=milestone_features_df.loc[milestone_network["to"]],
    )
    progressions = milestone_network.iloc[proj["segment"] - 1][["from", "to"]].copy()
    progressions["cell_id"] = fadata.obs.index
    progressions["percentage"] = proj["progression"]
    progressions = progressions[["cell_id", "from", "to", "percentage"]].reset_index(drop=True)

    final_graph = nx.from_pandas_edgelist(milestone_network, source="from", target="to", create_using=nx.DiGraph)
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

"""Waterfall-style two-level aggregation and time-based cell progressions."""

import numpy as np
import pandas as pd
from scipy import sparse


def _aggregate_time_maps(fadata, tmaps, time_key, cluster_key):
    """Pool row-normalized cell support at temporal and global type levels.

    This follows Waterfall waterfall/tl/milestone.py. Raw transport mass is
    retained separately for Sankey plots; support counts matched source cells.
    """
    # ========== Step 1: Read and validate times and cell types ==========
    obs = fadata.obs
    times = pd.to_numeric(obs[time_key]).to_numpy(dtype=float)
    if not np.isfinite(times).all() or obs[cluster_key].isna().any():
        raise ValueError("Times must be finite and state labels must not be missing.")
    if not tmaps or not len(obs):
        raise ValueError("timezone requires cells and at least one time-pair matrix.")
    groups = obs[cluster_key].astype(str).to_numpy()
    order, time_index = np.unique(times, return_inverse=True)
    rank = {time: index for index, time in enumerate(order)}

    # ========== Step 2: Assemble adjacent-time tmaps into the global weight matrix W ==========
    # Map local tmap row/column indices to global obs indices.
    rows, cols, values = [], [], []
    for (t1, t2), tmap in tmaps.items():
        t1, t2 = float(t1), float(t2)
        if t1 not in rank or t2 not in rank or t2 <= t1:
            raise ValueError("Transport pairs must join observed times in forward order.")
        if rank[t2] != rank[t1] + 1:
            raise ValueError("timezone requires adjacent observation time pairs.")
        source, target = np.flatnonzero(times == t1), np.flatnonzero(times == t2)
        matrix = sparse.coo_matrix(tmap, dtype=float, copy=True)
        if matrix.shape != (len(source), len(target)):
            raise ValueError(f"Transport shape {matrix.shape} differs from {(len(source), len(target))}.")
        if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError("Transport weights must be finite and nonnegative.")
        rows.extend(source[matrix.row])
        cols.extend(target[matrix.col])
        values.extend(matrix.data)
    weights = sparse.csr_matrix((values, (rows, cols)), shape=(len(obs), len(obs)))
    weights.eliminate_zeros()

    # ========== Step 3: Row-normalize W into the conditional transition matrix P ==========
    # Each matched source cell contributes one support unit; unmatched rows stay zero.
    mass = np.asarray(weights.sum(axis=1)).ravel()
    inverse = np.divide(1.0, mass, out=np.zeros_like(mass), where=mass > 0)
    probabilities = sparse.diags(inverse) @ weights

    # ========== Step 4: Aggregate by time and type into temporal nodes and edges ==========
    # Count observed cells, nonterminal-time cells, and sources with outgoing support.
    cells = pd.DataFrame({"time": times, "group": groups, "eligible": time_index < len(order) - 1, "mapped": mass > 0})
    nodes = (
        cells.groupby(["time", "group"], sort=True)
        .agg(n_cells=("group", "size"), n_source_cells=("eligible", "sum"), n_mapped_cells=("mapped", "sum"))
        .reset_index()
    )
    nodes.insert(0, "node_id", [f"state_{i}" for i in range(len(nodes))])
    membership = pd.MultiIndex.from_frame(nodes[["time", "group"]]).get_indexer(pd.MultiIndex.from_arrays([times, groups]))
    # Q assigns cells to temporal states: Q.T @ P @ Q pools support; Q.T @ W @ Q preserves raw OT mass.
    assignment = sparse.csr_matrix((np.ones(len(obs)), (np.arange(len(obs)), membership)), shape=(len(obs), len(nodes)))
    support = (assignment.T @ probabilities @ assignment).tocoo()
    raw_mass = (assignment.T @ weights @ assignment).tocsr()
    source, target = nodes.iloc[support.row], nodes.iloc[support.col]
    edges = (
        pd.DataFrame(
            {
                "source": source["node_id"].to_numpy(),
                "target": target["node_id"].to_numpy(),
                "source_time": source["time"].to_numpy(),
                "target_time": target["time"].to_numpy(),
                "source_group": source["group"].to_numpy(),
                "target_group": target["group"].to_numpy(),
                "support": support.data,
                "raw_support": np.asarray(raw_mass[support.row, support.col]).ravel() if support.nnz else np.empty(0),
                "n_source_cells": source["n_source_cells"].to_numpy(),
                "n_mapped_cells": source["n_mapped_cells"].to_numpy(),
            }
        )
        .sort_values(["source_time", "source_group", "target_group"])
        .reset_index(drop=True)
    )
    edges["transition_fraction"] = edges["support"] / edges.groupby("source")["support"].transform("sum")

    # ========== Step 5: Merge same-type nodes across times into global milestone tables ==========
    # Pool support before normalization; do not sum per-window fractions.
    global_nodes = (
        nodes.groupby("group", sort=True)
        .agg(n_cells=("n_cells", "sum"), n_source_cells=("n_source_cells", "sum"), n_mapped_cells=("n_mapped_cells", "sum"))
        .reset_index()
        .rename(columns={"group": "node_id"})
    )
    global_nodes["group"] = global_nodes["node_id"]
    global_edges = (
        edges.groupby(["source_group", "target_group"], sort=True)
        .agg(support=("support", "sum"), n_windows=("source_time", "nunique"))
        .reset_index()
        .rename(columns={"source_group": "source", "target_group": "target"})
    )
    # Include same-type persistence in the denominator; filter displayed self-loops later.
    global_edges["transition_fraction"] = global_edges["support"] / global_edges.groupby("source")["support"].transform("sum")
    counts = global_nodes.set_index("node_id")
    for column in ["n_source_cells", "n_mapped_cells"]:
        global_edges[column] = global_edges["source"].map(counts[column]).astype(int)
    global_edges["is_self_loop"] = global_edges["source"] == global_edges["target"]
    tables = dict(temporal_nodes=nodes, temporal_edges=edges, global_nodes=global_nodes, global_edges=global_edges)
    for table in tables.values():
        table.index = table.index.astype(str)
    return tables, probabilities.tocsr(), times, groups


def _first_type_change(probabilities, times, groups, states, backward=False):
    """Propagate through same-type cells to the first different type.

    Forward scores identify future branches; reverse conditional scores locate
    parent branches of terminal cells. Neither requires a verified lineage.
    """
    matrix = probabilities
    if backward:
        # Row-normalize the transpose for predecessor support; this is not a matrix inverse.
        matrix = matrix.T.tocsr()
        mass = np.asarray(matrix.sum(axis=1)).ravel()
        inverse = np.divide(1.0, mass, out=np.zeros_like(mass), where=mass > 0)
        matrix = sparse.diags(inverse) @ matrix
    codes = pd.Index(states).get_indexer(groups)
    # scores[i, k] is the support for type k at cell i's first type change along matched paths.
    scores = np.zeros((len(times), len(states)))
    # Traverse reverse time for future scores and chronological time for predecessor scores to resolve dependencies.
    order = np.unique(times) if backward else np.unique(times)[::-1]
    for time in order:
        indices = np.flatnonzero(times == time)
        block = matrix[indices].tocoo()
        same = groups[indices[block.row]] == groups[block.col]
        # Propagate through same-type edges; stop at the first different type and credit its support.
        persistence = sparse.csr_matrix((block.data[same], (block.row[same], block.col[same])), shape=block.shape)
        changes = sparse.csr_matrix((block.data[~same], (block.row[~same], codes[block.col[~same]])), shape=(len(indices), len(states)))
        scores[indices] = persistence @ scores + changes.toarray()
    return scores


def build_trajectory_timezone(
    fadata,
    tmaps: dict,
    time_key: str = None,
    cluster_key: str = None,
    flow_threshold: float = 0.1,
    relative_threshold: float = 0.3,
    normalize: bool = True,
    include_self_loop: bool = False,
    cell_time_key: str = None,
):
    """Merge time-state evidence into type milestones and time-based progressions.

    Each matched source contributes one support unit (Waterfall semantics, not
    absolute OT flux). Global fractions pool support before normalization and
    include persistence in their denominator. Thresholds act on global fractions
    when normalize=True, otherwise on support; full evidence is kept in metadata.

    Cells select the first different future type along their tmap paths, or a
    previous type if no eligible future branch exists. Progress is linear from
    the earliest source-type observation to the latest target-type observation.
    This is an explicit display convention, not inferred differentiation age.
    Unmatched/pruned cells stay at their type node. No DAG/lineage is assumed.
    """
    # ========== Step 1: Read priors and build temporal and merged type support tables ==========
    prior = getattr(fadata, "prior_information", {})
    time_key = time_key or prior.get("time_key", "time")
    cluster_key = cluster_key or prior.get("cluster", "clusters")
    if flow_threshold < 0 or relative_threshold < 0:
        raise ValueError("Flow thresholds must be nonnegative.")
    tables, probabilities, times, groups = _aggregate_time_maps(fadata, tmaps, time_key, cluster_key)
    clocks = pd.to_numeric(fadata.obs[cell_time_key]).to_numpy(dtype=float) if cell_time_key else times
    if not np.isfinite(clocks).all() or np.any((clocks < times.min()) | (clocks > times.max())):
        raise ValueError("Cell time is outside the observed time range or is not finite.")

    # ========== Step 2: Filter global differentiation edges by absolute and relative thresholds ==========
    # normalize selects the filtering score; cell-level support is always source-normalized.
    edges = tables["global_edges"]
    scores = edges["transition_fraction"] if normalize else edges["support"]
    maximum = scores.groupby(edges["source"]).transform("max")
    edges["retained"] = (scores >= flow_threshold) & (scores >= relative_threshold * maximum)
    if not include_self_loop:
        edges["retained"] &= ~edges["is_self_loop"]

    # ========== Step 3: Build milestone_network and record eligible branches and time bounds ==========
    states = tables["global_nodes"]["node_id"].tolist()
    codes = pd.Index(states).get_indexer(groups)
    eligible = np.zeros((len(states), len(states)), dtype=bool)
    bounds = {}
    network = []
    # Bounds span the earliest source-type observation to the latest target-type observation, not one time window.
    for edge in edges.loc[edges["retained"]].itertuples():
        start, end = times[groups == edge.source].min(), times[groups == edge.target].max()
        network.append({"from": edge.source, "to": edge.target, "length": max(float(end - start), 1e-6), "directed": True})
        if not edge.is_self_loop:
            a, b = states.index(edge.source), states.index(edge.target)
            eligible[a, b] = True
            bounds[edge.source, edge.target] = start, end
    # Isolated type nodes require anchor edges in Cafe, not biological self-flow.
    used = {row[key] for row in network for key in ["from", "to"]}
    network.extend({"from": state, "to": state, "length": 1.0, "directed": True} for state in states if state not in used)

    # ========== Step 4: Trace first type changes along matched paths for future and predecessor support ==========
    future = _first_type_change(probabilities, times, groups, states)
    past = _first_type_change(probabilities, times, groups, states, backward=True)

    # ========== Step 5: Assign cells to differentiation edges and compute time-based progressions ==========
    positions = []
    for index, group in enumerate(groups):
        # Use retained edges only: prefer a future branch, then a predecessor; otherwise stay at the type node.
        forward = np.where(eligible[codes[index]], future[index], 0)
        reverse = np.where(eligible[:, codes[index]], past[index], 0)
        if forward.max() > 0:
            source, target = group, states[int(forward.argmax())]
            confidence, status = float(forward.max()), "outgoing"
        elif reverse.max() > 0:
            source, target = states[int(reverse.argmax())], group
            confidence, status = float(reverse.max()), "incoming"
        else:
            source = target = group
            confidence, status = 0.0, "unmatched"
        progress = 0.0
        if source != target:
            start, end = bounds[source, target]
            # Clip p=(t-start)/(end-start) to [0, 1]; this is a time position, not a fate probability.
            progress = float(np.clip((clocks[index] - start) / (end - start), 0, 1))
        positions.append([fadata.obs.index[index], source, target, progress, confidence, status])
    positions = pd.DataFrame(positions, columns=["cell_id", "from", "to", "percentage", "confidence", "status"])

    # ========== Step 6: Return the trajectory with support tables and cell positions for inspection ==========
    tables.update(
        cell_positions=positions, time_key=time_key, cluster_key=cluster_key, cell_time_key=cell_time_key or time_key, normalization="per_source_cell"
    )
    result = {
        "milestone_network": pd.DataFrame(network),
        "progressions": positions[["cell_id", "from", "to", "percentage"]].copy(),
        "raw_wrapper_dict": tables,
    }
    # Reuse cell-type colors for merged milestones to match the scatter plots.
    color_key = f"{cluster_key}_colors"
    if color_key in fadata.uns and isinstance(fadata.obs[cluster_key].dtype, pd.CategoricalDtype):
        result["milestone_color_dict"] = dict(zip(fadata.obs[cluster_key].cat.categories.astype(str), fadata.uns[color_key]))
    return result

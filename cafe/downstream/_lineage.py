from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from ..data import FateAnnData, MilestoneWrapper


@dataclass(frozen=True)
class Lineage:
    """Cell memberships and trajectory metadata for terminal lineages."""

    membership: pd.DataFrame
    exclusive_membership: pd.DataFrame
    pseudotime: pd.DataFrame
    subgraphs: Mapping[Hashable, nx.DiGraph]
    names: tuple[Hashable, ...]
    colors: tuple[str, ...]
    start_milestone: Hashable
    model_name: str

    @property
    def obs_names(self) -> pd.Index:
        """Observation names aligned with all cell-level matrices."""
        return self.membership.index


def _validate_graph(milestone_wrapper: MilestoneWrapper, start_milestone: Hashable) -> nx.DiGraph:
    milestone_network = milestone_wrapper.milestone_network
    if not milestone_network["directed"].all():
        raise ValueError("Lineage extraction requires a directed milestone graph.")

    graph = milestone_wrapper.milestone_network_G
    if not isinstance(graph, nx.DiGraph):
        raise ValueError("Lineage extraction requires a directed milestone graph.")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Lineage extraction requires an acyclic milestone graph.")
    if start_milestone not in graph:
        raise KeyError(f"Start milestone {start_milestone!r} is not present in the milestone graph.")
    return graph


def _resolve_terminal_states(
    graph: nx.DiGraph,
    start_milestone: Hashable,
    terminal_states: Sequence[Hashable] | None,
) -> tuple[Hashable, ...]:
    available = tuple(node for node in graph if node != start_milestone and nx.has_path(graph, start_milestone, node) and graph.out_degree(node) == 0)
    if terminal_states is None:
        selected = available
    else:
        selected = tuple(terminal_states)
        if len(set(selected)) != len(selected):
            raise ValueError("Terminal states must be unique.")
        invalid = [terminal for terminal in selected if terminal not in available]
        if invalid:
            raise ValueError(
                f"Selected states {invalid!r} are not reachable terminal milestones. " f"Available terminal milestones are {list(available)!r}."
            )

    if not selected:
        raise ValueError(f"No reachable terminal milestones were found from {start_milestone!r}.")
    return selected


def _terminal_subgraph(graph: nx.DiGraph, start_milestone: Hashable, terminal_state: Hashable) -> nx.DiGraph:
    nodes = {node for node in graph if nx.has_path(graph, start_milestone, node) and nx.has_path(graph, node, terminal_state)}
    return graph.subgraph(nodes).copy()


def _validate_cell_ids(fadata: FateAnnData, milestone_wrapper: MilestoneWrapper) -> None:
    if not fadata.obs_names.is_unique:
        raise ValueError("FateAnnData observation names must be unique.")

    progression_ids = pd.Index(milestone_wrapper.progressions["cell_id"].unique())
    missing = progression_ids.difference(fadata.obs_names)
    if len(missing):
        raise ValueError(f"Trajectory contains cell IDs missing from FateAnnData: {missing.tolist()!r}.")


def _membership_from_progressions(
    obs_names: pd.Index,
    progressions: pd.DataFrame,
    subgraphs: Mapping[Hashable, nx.DiGraph],
) -> pd.DataFrame:
    membership = pd.DataFrame(False, index=obs_names.copy(), columns=list(subgraphs), dtype=bool)

    for terminal_state, graph in subgraphs.items():
        edges = set(graph.edges)
        nodes = set(graph.nodes)
        selected = progressions.apply(
            lambda row, nodes=nodes, edges=edges: (row["from"] == row["to"] and row["from"] in nodes) or (row["from"], row["to"]) in edges,
            axis=1,
        )
        cell_ids = progressions.loc[selected, "cell_id"].unique()
        membership.loc[cell_ids, terminal_state] = True
        if not membership[terminal_state].any():
            raise ValueError(f"Terminal lineage {terminal_state!r} has no associated cells.")

    return membership


def _exclusive_membership(membership: pd.DataFrame) -> pd.DataFrame:
    unique = membership.sum(axis=1).eq(1)
    return membership.mul(unique, axis=0).astype(bool)


def _lineage_pseudotime(
    milestone_wrapper: MilestoneWrapper,
    membership: pd.DataFrame,
    subgraphs: Mapping[Hashable, nx.DiGraph],
    start_milestone: Hashable,
) -> pd.DataFrame:
    pseudotime = pd.DataFrame(np.nan, index=membership.index.copy(), columns=membership.columns, dtype=float)
    percentages = milestone_wrapper.milestone_percentages

    for terminal_state, graph in subgraphs.items():
        distances = nx.single_source_dijkstra_path_length(graph, start_milestone, weight="length")
        member_ids = membership.index[membership[terminal_state]]
        lineage_percentages = percentages[percentages["cell_id"].isin(member_ids) & percentages["milestone_id"].isin(distances)]

        values = (
            lineage_percentages.assign(
                weighted_distance=lambda frame, distances=distances: frame["percentage"] * frame["milestone_id"].map(distances)
            )
            .groupby("cell_id")["weighted_distance"]
            .sum()
        )
        values = values.reindex(member_ids)

        valid = values.notna()
        if valid.any():
            minimum = values[valid].min()
            maximum = values[valid].max()
            values.loc[valid] = 0.0 if maximum == minimum else (values[valid] - minimum) / (maximum - minimum)
        pseudotime.loc[member_ids, terminal_state] = values

    return pseudotime


def extract_lineages(
    fadata: FateAnnData,
    *,
    model_name: str | None = None,
    start_milestone: Hashable | None = None,
    start_cell: str | None = None,
    terminal_states: Sequence[Hashable] | None = None,
) -> Lineage:
    """Extract terminal lineages from a rooted directed tree or DAG."""
    parsed_model_name = fadata.parse_model_name(model_name)
    if parsed_model_name is None:
        raise KeyError(f"Trajectory model {model_name!r} is not available.")

    milestone_wrapper = fadata.get_milestone_wrapper(parsed_model_name)
    if milestone_wrapper is None:
        raise ValueError(f"Trajectory model {parsed_model_name!r} does not contain a MilestoneWrapper.")

    resolved_start = fadata._check_start_milestone(
        start_milestone=start_milestone,
        start_cell=start_cell,
        model_name=parsed_model_name,
    )
    graph = _validate_graph(milestone_wrapper, resolved_start)
    selected_terminals = _resolve_terminal_states(graph, resolved_start, terminal_states)
    _validate_cell_ids(fadata, milestone_wrapper)

    subgraphs = {terminal_state: _terminal_subgraph(graph, resolved_start, terminal_state) for terminal_state in selected_terminals}
    membership = _membership_from_progressions(fadata.obs_names, milestone_wrapper.progressions, subgraphs)
    exclusive_membership = _exclusive_membership(membership)
    pseudotime = _lineage_pseudotime(milestone_wrapper, membership, subgraphs, resolved_start)

    color_map = milestone_wrapper.milestone_color_dict
    colors = tuple(color_map.get(terminal_state, "#808080") for terminal_state in selected_terminals)
    return Lineage(
        membership=membership,
        exclusive_membership=exclusive_membership,
        pseudotime=pseudotime,
        subgraphs=subgraphs,
        names=selected_terminals,
        colors=colors,
        start_milestone=resolved_start,
        model_name=parsed_model_name,
    )

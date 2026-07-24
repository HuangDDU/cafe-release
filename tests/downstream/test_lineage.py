import networkx as nx
import numpy as np
import pandas as pd
import pytest

import cafe
from cafe.downstream._lineage import extract_lineages


def make_fadata(milestone_network, progressions, model_name="test"):
    obs_names = pd.Index(progressions["cell_id"].unique())
    fadata = cafe.data.FateAnnData(
        X=np.zeros((len(obs_names), 2)),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    fadata.add_model_name(model_name)
    fadata.add_trajectory(
        milestone_network=milestone_network,
        progressions=progressions,
        generate_color=False,
    )
    return fadata


def make_tree_fadata():
    milestone_network = pd.DataFrame(
        {
            "from": ["Root", "Branch", "Branch"],
            "to": ["Branch", "Alpha", "Beta"],
            "length": [1.0, 1.0, 1.0],
            "directed": [True, True, True],
        }
    )
    progressions = pd.DataFrame(
        {
            "cell_id": ["root", "trunk", "alpha_1", "alpha_2", "beta_1", "beta_2"],
            "from": ["Root", "Root", "Branch", "Alpha", "Branch", "Beta"],
            "to": ["Root", "Branch", "Alpha", "Alpha", "Beta", "Beta"],
            "percentage": [1.0, 0.5, 0.5, 1.0, 0.5, 1.0],
        }
    )
    return make_fadata(milestone_network, progressions, model_name="tree")


def test_extract_tree_lineages_preserves_shared_and_exclusive_membership():
    fadata = make_tree_fadata()

    lineages = extract_lineages(
        fadata,
        model_name="tree",
        start_milestone="Root",
        terminal_states=["Alpha", "Beta"],
    )

    assert lineages.names == ("Alpha", "Beta")
    assert lineages.membership.loc["trunk"].tolist() == [True, True]
    assert lineages.exclusive_membership.loc["trunk"].tolist() == [False, False]
    assert lineages.exclusive_membership.loc["alpha_1"].tolist() == [True, False]
    assert lineages.exclusive_membership.loc["beta_1"].tolist() == [False, True]
    assert lineages.pseudotime.loc["alpha_2", "Alpha"] == pytest.approx(1.0)
    assert np.isnan(lineages.pseudotime.loc["beta_2", "Alpha"])


def test_extract_dag_lineage_keeps_all_routes_to_terminal():
    milestone_network = pd.DataFrame(
        {
            "from": ["Root", "Root", "A", "B", "Merge"],
            "to": ["A", "B", "Merge", "Merge", "End"],
            "length": [1.0] * 5,
            "directed": [True] * 5,
        }
    )
    progressions = pd.DataFrame(
        {
            "cell_id": ["a", "b", "merge", "end"],
            "from": ["Root", "Root", "A", "Merge"],
            "to": ["A", "B", "Merge", "End"],
            "percentage": [0.5, 0.5, 0.5, 1.0],
        }
    )
    fadata = make_fadata(milestone_network, progressions, model_name="dag")

    lineages = extract_lineages(
        fadata,
        model_name="dag",
        start_milestone="Root",
        terminal_states=["End"],
    )

    graph = lineages.subgraphs["End"]
    assert isinstance(graph, nx.DiGraph)
    assert set(graph.edges) == {
        ("Root", "A"),
        ("Root", "B"),
        ("A", "Merge"),
        ("B", "Merge"),
        ("Merge", "End"),
    }


def test_extract_lineages_selects_reachable_sinks_by_default():
    fadata = make_tree_fadata()

    lineages = extract_lineages(fadata, model_name="tree", start_milestone="Root")

    assert lineages.names == ("Alpha", "Beta")


@pytest.mark.parametrize(
    ("network", "match"),
    [
        (
            pd.DataFrame(
                {
                    "from": ["Root"],
                    "to": ["End"],
                    "length": [1.0],
                    "directed": [False],
                }
            ),
            "directed",
        ),
        (
            pd.DataFrame(
                {
                    "from": ["Root", "End"],
                    "to": ["End", "Root"],
                    "length": [1.0, 1.0],
                    "directed": [True, True],
                }
            ),
            "acyclic",
        ),
    ],
)
def test_extract_lineages_rejects_unsupported_graphs(network, match):
    progressions = pd.DataFrame(
        {
            "cell_id": ["cell"],
            "from": ["Root"],
            "to": ["End"],
            "percentage": [0.5],
        }
    )
    fadata = make_fadata(network, progressions)

    with pytest.raises(ValueError, match=match):
        extract_lineages(fadata, model_name="test", start_milestone="Root")


def test_extract_lineages_rejects_non_terminal_target():
    fadata = make_tree_fadata()

    with pytest.raises(ValueError, match="terminal"):
        extract_lineages(
            fadata,
            model_name="tree",
            start_milestone="Root",
            terminal_states=["Branch"],
        )

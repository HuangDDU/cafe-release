import numpy as np
import pandas as pd

from .fate_anndata import FateAnnData


topologies_with_same_n_milestones = {
    "linear": pd.DataFrame({
        "from": ["A", "B", "C", "D", "E"],
        "to": ["B", "C", "D", "E", "F"],
    }),
    "bifurcation": pd.DataFrame({
        "from": ["A", "B", "B", "C", "D"],
        "to": ["B", "C", "D", "E", "F"],
    }),
    "multifurcating": pd.DataFrame({
        "from": ["A", "B", "B", "B", "C"],
        "to": ["B", "C", "D", "E", "F"],
    }),
    "tree": pd.DataFrame({
        "from": ["A", "B", "B", "C", "C"],
        "to": ["B", "C", "D", "E", "F"],
    }),
    "cycle": pd.DataFrame({
        "from": ["A", "B", "C", "D", "E", "F"],
        "to": ["B", "C", "D", "E", "F", "A"],
    }),
    "connected": pd.DataFrame({
        "from": ["A", "B", "B", "D", "E", "C"],
        "to": ["B", "C", "D", "E", "A", "F"],
    }),
    "disconnected": pd.DataFrame({
        "from": ["A", "B", "C", "D", "E"],
        "to": ["B", "C", "A", "E", "F"],
    })
}


def generate_trajectory(milestone_network: pd.DataFrame, id: str = ""):
    # ref: PyDynverse/pydynverse/toy/generate_trajectory.py
    if "length" not in milestone_network.columns:
        milestone_network["length"] = 1
    if "directed" not in milestone_network.columns:
        milestone_network["directed"] = True

    # only 1x1 expression matrix
    fadata = FateAnnData(X=np.zeros((1, 1)))
    fadata.obs.index = ["a"]

    divergence_regions = pd.DataFrame(columns=["divergence_id", "milestone_id", "is_star"])
    # milestone_percentages = pd.DataFrame(
    #     data=[["a", milestone_network.iloc[0, 0], 1.0],],
    #     columns=["cell_id", "milestone_id", "percentage"])
    # fadata.add_trajectory(milestone_network, divergence_regions, milestone_percentages)
    from_milestone, to_milestone = milestone_network["from"][0], milestone_network["to"][0]
    progressions = pd.DataFrame(
        columns=["cell_id", "from", "to", "percentage"],
        data=[["a", from_milestone, to_milestone, 0.5]],
    )
    fadata.add_trajectory(
        milestone_network=milestone_network,
        divergence_regions=divergence_regions,
        progressions=progressions)

    return fadata

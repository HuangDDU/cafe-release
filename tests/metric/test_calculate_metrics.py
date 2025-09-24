import numpy as np
import pandas as pd
import pytest

import cfe
import cfe.metric


def test_calculate_metrics():
    cell_ids = ["a", "b", "c", "d", "e"]
    # milestone_ids = ["A", "B", "C", "D"]
    fadata = cfe.data.FateAnnData(name=id, X=np.zeros((len(cell_ids), 2)))

    # trajectory1: ref model
    fadata.add_model_name("ref")
    milestone_network = pd.DataFrame(
        data=[["A", "B", 1, True], ["B", "C", 1, True], ["C", "D", 1, True]],
        columns=["from", "to", "length", "directed"],
    )
    progressions = pd.DataFrame(
        data=[
            ["a", "A", "B", 0.3],
            ["b", "A", "B", 0.6],
            ["c", "B", "C", 0.2],
            ["d", "B", "C", 0.8],
            ["e", "C", "D", 0.4],
        ],
        columns=["cell_id", "from", "to", "percentage"],
    )
    fadata.add_trajectory(milestone_network=milestone_network, progressions=progressions)

    # trajectory2: new model
    fadata.add_model_name("new")
    milestone_network = pd.DataFrame(
        data=[["A", "B", 1, True], ["B", "C", 1, True], ["B", "D", 1, True]],
        columns=["from", "to", "length", "directed"],
    )
    progressions = pd.DataFrame(
        data=[
            ["a", "A", "B", 0.3],
            ["b", "A", "B", 0.7],
            ["c", "B", "C", 0.2],
            ["d", "B", "C", 0.8],
            ["e", "B", "D", 0.5],
        ],
        columns=["cell_id", "from", "to", "percentage"],
    )
    fadata.add_trajectory(milestone_network=milestone_network, progressions=progressions)

    summary_dict_self = cfe.metric.calculate_metrics(fadata, now_model="new", ref_model="new")  # calculate metric with self, max metric value
    summary_dict = cfe.metric.calculate_metrics(fadata, now_model="new", ref_model="ref")  # calculate metric with ref

    assert summary_dict_self["isomorphic"] == 1
    assert summary_dict["isomorphic"] == 0

    assert summary_dict_self["edge_flip"] == 1
    assert summary_dict["edge_flip"] == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])

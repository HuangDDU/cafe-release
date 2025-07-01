import numpy as np
import pandas as pd
import pytest

import cfe


def test_project_to_segments():
    # input data
    # reuse data from data.test_fate_anndata.test_add_trajectory_projection
    x = pd.DataFrame(
        columns=["cell_id", "comp_1", "comp_2"],
        data=[
            ["a", 0, 1.5],
            ["b", 0.8, 0.5],
            ["c", 1.2, 0.5],
            ["d", 2, 0.5],
            ["e", 1.2, 1.5],
            ["f", 2.4, 1.5],
        ],
    )
    segment_start = pd.DataFrame(
        columns=["milestone_id", "comp_1", "comp_2"],
        data=[
            ["W", 0, 1],
            ["X", 1, 1],
            ["X", 1, 1],
            ["Z", 2, 1],
        ],
    )
    segment_end = pd.DataFrame(
        columns=["milestone_id", "comp_1", "comp_2"],
        data=[
            ["X", 1, 1],
            ["Y", 1, 2],
            ["Z", 2, 1],
            ["A", 4, 1],
        ],
    )
    x.set_index("cell_id", inplace=True)
    segment_start.set_index("milestone_id", inplace=True)
    segment_end.set_index("milestone_id", inplace=True)

    # execute function
    out = cfe.util.project_to_segments(x, segment_start, segment_end)

    # expected result
    expected_x_proj = (
        pd.DataFrame(
            columns=["cell_id", "comp_1", "comp_2"],
            data=[
                ["a", 0.0, 1.0],
                ["b", 0.8, 1.0],
                ["c", 1.2, 1.0],
                ["d", 2.0, 1.0],
                ["e", 1.0, 1.5],
                ["f", 2.4, 1.0],
            ],
        )
        .set_index("cell_id", drop=True)
        .values
    )  # projection points
    expected_distance = np.array([0.25, 0.25, 0.25, 0.25, 0.04, 0.25])  # squred projection dist
    expected_segment = np.array([1, 1, 3, 3, 2, 4])  # the index of projected segment
    expected_progression = np.array([0, 0.8, 0.2, 1, 0.5, 0.2])  # 投影点在所在边额比例

    # assert
    assert (out["x_proj"] == expected_x_proj).any()
    assert (out["distance"] == expected_distance).any()
    assert (out["segment"] == expected_segment).any()
    assert (out["progression"] == expected_progression).any()


if __name__ == "__main__":
    pytest.main(["-v", __file__])

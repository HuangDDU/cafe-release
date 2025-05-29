import pytest
import cfe

import pandas as pd
from ..test_util import compare_dataframes, compare_dataframes_closely


class TestWaypointWrapper:
    def setup_method(self):
        from .test_fate_milestone_wrapper import setup_method_data
        self.milestone_wrapper = setup_method_data()
        self.waypoint_wrapper = cfe.data.WaypointWrapper(self.milestone_wrapper, resolution=1)

    def test_magic_method(self):
        """test __***__ methods"""
        ww = self.waypoint_wrapper

        # test __contains__
        assert "id" in ww, "id should in mw"

        # test __getitem__
        assert ww["id"] == ww.id, "mw['id'] should be the same as mw.id"

        # test keys
        ww_dict = dict(ww)
        attribute_name_list = ["id"]
        assert set(attribute_name_list).issubset(set(ww.keys())), f"{attribute_name_list} should be the keys of the dict: {ww_dict}"

    def test_init(self):
        ww = self.waypoint_wrapper
        # ww.pipeline()

        assert ww.waypoint_milestone_percentages is not None
        assert ww.waypoint_progressions is not None
        assert ww.waypoint_geodesic_distances is not None
        assert ww.waypoint_network is not None
        assert ww.waypoints is not None

    def test_select_waypoints(self):
        ww = self.waypoint_wrapper
        # ww._select_waypoints(resolution=1)

        # 预期构造结果
        expected_waypoint_milestone_percentages = pd.DataFrame(
            columns=["waypoint_id", "milestone_id", "percentage"],
            data=[
                ["MILESTONE_BEGIN_WW_X", "W", 1],
                ["MILESTONE_BEGIN_WW_X", "X", 0],
                ["MILESTONE_BEGIN_WX_Y", "X", 1],
                ["MILESTONE_BEGIN_WX_Y", "Y", 0],
                ["MILESTONE_BEGIN_WX_Z", "X", 1],
                ["MILESTONE_BEGIN_WX_Z", "Z", 0],
                ["MILESTONE_BEGIN_WZ_A", "Z", 1],
                ["MILESTONE_BEGIN_WZ_A", "A", 0],
                ["MILESTONE_END_WW_X", "W", 0],
                ["MILESTONE_END_WW_X", "X", 1],
                ["MILESTONE_END_WX_Y", "X", 0],
                ["MILESTONE_END_WX_Y", "Y", 1],
                ["MILESTONE_END_WX_Z", "X", 0],
                ["MILESTONE_END_WX_Z", "Z", 1],
                ["MILESTONE_END_WZ_A", "Z", 0],
                ["MILESTONE_END_WZ_A", "A", 1],
                ["W8", "Z", 0.5],
                ["W8", "A", 0.5],
            ]
        )
        expected_waypoint_progressions = pd.DataFrame(
            columns=["from", "to", "percentage", "waypoint_id"],
            data=[
                ["W", "X", 0, "MILESTONE_BEGIN_WW_X"],
                ["W", "X", 1, "MILESTONE_END_WW_X"],
                ["X", "Y", 0, "MILESTONE_BEGIN_WX_Y"],
                ["X", "Y", 1, "MILESTONE_END_WX_Y"],
                ["X", "Z", 0, "MILESTONE_BEGIN_WX_Z"],
                ["X", "Z", 1, "MILESTONE_END_WX_Z"],
                ["Z", "A", 0, "MILESTONE_BEGIN_WZ_A"],
                ["Z", "A", 0.5, "W8"],
                ["Z", "A", 1, "MILESTONE_END_WZ_A"],
            ]
        )
        expected_waypoint_geodesic_distances = pd.DataFrame(
            columns=self.milestone_wrapper.cell_id_list,
            index=["MILESTONE_BEGIN_WW_X",
                   "MILESTONE_END_WW_X",
                   "MILESTONE_BEGIN_WX_Y",
                   "MILESTONE_END_WX_Y",
                   "MILESTONE_BEGIN_WX_Z",
                   "MILESTONE_END_WX_Z",
                   "MILESTONE_BEGIN_WZ_A",
                   "W8",
                   "MILESTONE_END_WZ_A",],
            data=[
                [0.0, 0.8, 1.2, 2.0, 1.7, 2.4],
                [1.0, 0.2, 0.2, 1.0, 0.7, 1.4],
                [1.0, 0.2, 0.2, 1.0, 0.7, 1.4],
                [2.0, 1.2, 1.2, 2.0, 1.3, 2.4],
                [1.0, 0.2, 0.2, 1.0, 0.7, 1.4],
                [2.0, 1.2, 0.8, 0.0, 0.7, 0.4],
                [2.0, 1.2, 0.8, 0.0, 0.7, 0.4],
                [3.0, 2.2, 1.8, 1.0, 1.7, 0.6],
                [4.0, 3.2, 2.8, 2.0, 2.7, 1.6],
            ]
        )
        expected_waypoint_network = pd.DataFrame(
            columns=["from", "to", "from_milestone_id", "to_milestone_id"],
            data=[
                ["MILESTONE_BEGIN_WW_X", "MILESTONE_END_WW_X", "W", "X"],
                ["MILESTONE_BEGIN_WX_Y", "MILESTONE_END_WX_Y", "X", "Y"],
                ["MILESTONE_BEGIN_WX_Z", "MILESTONE_END_WX_Z", "X", "Z"],
                ["MILESTONE_BEGIN_WZ_A", "W8", "Z", "A"],
                ["W8", "MILESTONE_END_WZ_A", "Z", "A"],
            ]
        )
        expected_waypoints = pd.DataFrame(
            columns=["waypoint_id", "milestone_id"],
            data=[
                ["MILESTONE_BEGIN_WW_X", "W"],
                ["MILESTONE_BEGIN_WX_Y", "X"],
                ["MILESTONE_BEGIN_WX_Z", "X"],
                ["MILESTONE_BEGIN_WZ_A", "Z"],
                ["MILESTONE_END_WW_X", "X"],
                ["MILESTONE_END_WX_Y", "Y"],
                ["MILESTONE_END_WX_Z", "Z"],
                ["MILESTONE_END_WZ_A", "A"],
                ["W8", None],
            ]
        )

        assert compare_dataframes(ww.waypoint_milestone_percentages, expected_waypoint_milestone_percentages, ["waypoint_id", "milestone_id"])
        assert compare_dataframes(ww.waypoint_progressions, expected_waypoint_progressions, ["from", "to", "waypoint_id"])
        assert compare_dataframes_closely(ww.waypoint_geodesic_distances, expected_waypoint_geodesic_distances, expected_waypoint_geodesic_distances.columns.tolist())
        assert compare_dataframes(ww.waypoint_network, expected_waypoint_network, ["from", "to"])
        assert compare_dataframes(ww.waypoints, expected_waypoints, "waypoint_id")

    def calculate_geodesic_distances(self):
        # test in test_select_waypoints
        pass


if __name__ == "__main__":
    pytest.main(["-v", __file__])

import pandas as pd
import pytest

import cafe

from ..test_util import compare_dataframes, compare_dataframes_closely


def setup_method_data():
    """Create data for testing, convinient for other test file reuse"""
    # id_list = ["W", "X", "Y", "Z", "A"]
    milestone_network = pd.DataFrame(
        columns=["from", "to", "length", "directed"],
        data=[["W", "X", 1.0, True], ["X", "Y", 1.0, True], ["X", "Z", 1.0, True], ["Z", "A", 2.0, True]],
    )
    divergence_regions = pd.DataFrame(
        columns=["divergence_id", "milestone_id", "is_start"],
        data=[["XYZ", "X", True], ["XYZ", "Y", False], ["XYZ", "Z", False]],
    )
    milestone_percentages = pd.DataFrame(
        columns=["cell_id", "milestone_id", "percentage"],
        data=[
            ["a", "W", 1.0],
            ["b", "W", 0.2],
            ["b", "X", 0.8],
            ["c", "X", 0.8],
            ["c", "Z", 0.2],
            ["d", "Z", 1.0],
            ["e", "X", 0.3],
            ["e", "Y", 0.2],
            ["e", "Z", 0.5],
            ["f", "Z", 0.8],
            ["f", "A", 0.2],
        ],
    )
    milestone_wrapper = cafe.data.MilestoneWrapper(
        milestone_network=milestone_network,
        divergence_regions=divergence_regions,
        milestone_percentages=milestone_percentages,
    )
    return milestone_wrapper


# test data for "merge_milestone_trajectory"


def get_merge_milestone_trajectory_mw():
    # global: A -> B -> {C, D}
    global_milestone_network = pd.DataFrame(
        columns=["from", "to", "length", "directed"],
        data=[
            ["A", "B", 1.0, True],
            ["B", "C", 1.0, True],
            ["B", "D", 1.0, True],
        ],
    )
    global_progressions = pd.DataFrame(
        columns=["cell_id", "from", "to", "percentage"],
        data=[
            ["a", "A", "B", 0.2],
            ["b1", "A", "B", 0.8],
            ["b2", "B", "C", 0.2],
            ["b3", "B", "D", 0.2],
            ["c", "B", "C", 0.8],
            ["d", "B", "D", 0.8],
        ],
    )
    global_divergence_regions = pd.DataFrame(
        columns=["divergence_id", "milestone_id", "is_start"], data=[["BCD", "B", True], ["BCD", "C", False], ["BCD", "D", False]]
    )
    global_milestone_wrapper = cafe.data.MilestoneWrapper(
        milestone_network=global_milestone_network,
        divergence_regions=global_divergence_regions,
        progressions=global_progressions,
    )

    # local sub trajectory to replace B: B1 -> {B2, B3}
    local_milestone_network = pd.DataFrame(
        columns=["from", "to", "length", "directed"],
        data=[
            ["B1", "B2", 0.5, True],
            ["B1", "B3", 0.5, True],
        ],
    )
    local_progressions = pd.DataFrame(
        columns=["cell_id", "from", "to", "percentage"],
        data=[
            ["b1", "B1", "B2", 0.2],
            ["b1", "B1", "B3", 0.2],
            ["b2", "B1", "B2", 0.8],
            ["b3", "B1", "B3", 0.8],
        ],
    )
    local_divergence_regions = pd.DataFrame(
        columns=["divergence_id", "milestone_id", "is_start"], data=[["B1B2B3", "B1", True], ["B1B2B3", "B2", False], ["B1B2B3", "B3", False]]
    )
    local_milestone_wrapper = cafe.data.MilestoneWrapper(
        milestone_network=local_milestone_network,
        divergence_regions=local_divergence_regions,
        progressions=local_progressions,
    )
    replace_milestone = "B"

    # expected milestone wrapper after merging:
    merged_milestone_network = pd.DataFrame(
        columns=["from", "to", "length", "directed"],
        data=[
            ["A", "B1", 1.0, True],
            ["B1", "B2", 1.0, True],
            ["B1", "B3", 1.0, True],
            ["B2", "C", 1.0, True],
            ["B3", "D", 1.0, True],
        ],
    )
    merged_progressions = pd.DataFrame(
        columns=["cell_id", "from", "to", "percentage"],
        data=[
            ["a", "A", "B1", 0.2],
            ["b1", "B1", "B2", 0.2],
            ["b1", "B1", "B3", 0.2],
            ["b2", "B1", "B2", 0.8],
            ["b3", "B1", "B3", 0.8],
            ["c", "B2", "C", 0.8],
            ["d", "B3", "D", 0.8],
        ],
    )
    merged_divergence_regions = pd.DataFrame(
        columns=["divergence_id", "milestone_id", "is_start"], data=[["B1B2B3", "B1", True], ["B1B2B3", "B2", False], ["B1B2B3", "B3", False]]
    )
    merged_milestone_wrapper = cafe.data.MilestoneWrapper(
        milestone_network=merged_milestone_network,
        progressions=merged_progressions,
        divergence_regions=merged_divergence_regions,
    )
    data = {
        "global_milestone_wrapper": global_milestone_wrapper,
        "local_milestone_wrapper": local_milestone_wrapper,
        "merged_milestone_wrapper": merged_milestone_wrapper,
        "replace_milestone": replace_milestone,
    }
    return data


class TestMilestoneWrapper:
    def setup_method(self):
        self.milestone_wrapper = setup_method_data()

    def test_magic_method(self):
        """test __***__ methods"""
        mw = self.milestone_wrapper

        # test __contains__
        assert "id" in mw, "id should in mw"

        # test __getitem__
        assert mw["id"] == mw.id, "mw['id'] should be the same as mw.id"

        # test keys
        mw_dict = dict(mw)
        attribute_name_list = ["id"]
        assert set(attribute_name_list).issubset(set(mw_dict.keys())), f"{attribute_name_list} should be the keys of the dict: {mw_dict}"

    def test_milestone_network(self):
        mw = self.milestone_wrapper
        id_list = mw.id_list
        milestone_network = mw.milestone_network
        assert (set(milestone_network["from"].unique()) | set(milestone_network["to"].unique())) == set(
            id_list
        ), "every id should show in 'from' or 'to' column in  milestone_network dataframe"

    def test_convert_milestone_percentages_to_progressions(self):
        mw = self.milestone_wrapper
        # static method can be called with class or instance
        progression = mw.convert_milestone_percentages_to_progressions(mw.milestone_network, mw.milestone_percentages)

        expected_progression = pd.DataFrame(
            columns=["cell_id", "from", "to", "percentage"],
            data=[
                ["a", "W", "W", 1],
                ["b", "W", "X", 0.8],
                ["c", "X", "Z", 0.2],
                ["d", "Z", "Z", 1],
                ["e", "X", "Y", 0.2],
                ["e", "X", "Z", 0.5],
                ["f", "Z", "A", 0.2],
            ],
        )
        assert isinstance(progression, pd.DataFrame), "progression should be a dataframe"
        assert compare_dataframes(progression, expected_progression, on_columns=["cell_id", "from", "to"])

    # this test case can't execute the method
    # def test_convert_progressions_to_milestone_percentages(self):
    #     pass

    def test_convert_progressions_to_milestone_percentages(self):
        from cafe.data import MilestoneWrapper

        milestone_network = pd.DataFrame(
            columns=["from", "to", "length", "directed"],
            data=[
                ["milestone_begin", "A", 1, True],
                ["milestone_begin", "B", 1, True],
                ["milestone_begin", "C", 1, True],
            ],
        )
        end_state_probabilities = pd.DataFrame(
            columns=["cell_id", "A", "B", "C"],
            data=[
                ["a", 0.5, 0.2, 0.2],
                ["b", 0.2, 0.5, 0.2],
                ["c", 0.2, 0.2, 0.5],
            ],
        )
        progressions = end_state_probabilities.melt(id_vars=["cell_id"], var_name="to", value_name="percentage")
        progressions["from"] = "milestone_begin"

        milestone_percentages = MilestoneWrapper.convert_progressions_to_milestone_percentages(
            milestone_network=milestone_network,
            progressions=progressions,
        )

        expected_milestone_percentages = pd.DataFrame(
            columns=["cell_id", "milestone_id", "percentage"],
            data=[
                [
                    "a",
                    "milestone_begin",
                    0.1,
                ],  # for start milestone， percentage = 1 - sum(other end milestone percentages)
                ["a", "A", 0.5],
                ["a", "B", 0.2],
                ["a", "C", 0.2],
                ["b", "milestone_begin", 0.1],
                ["b", "A", 0.2],
                ["b", "B", 0.5],
                ["b", "C", 0.2],
                ["c", "milestone_begin", 0.1],
                ["c", "A", 0.2],
                ["c", "B", 0.2],
                ["c", "C", 0.5],
            ],
        )

        assert compare_dataframes_closely(milestone_percentages, expected_milestone_percentages, on_columns=["cell_id", "milestone_id"])

    def test_generate_color(self):
        mw = self.milestone_wrapper
        # lazy load for milestone_color_dict and cell_color_dict, automatic call function '_generate_color' when first call
        milestone_color_dict = mw.milestone_color_dict
        cell_color_dict = mw.cell_color_dict
        assert isinstance(milestone_color_dict, dict), "milestone_color_dict should be a dict"
        assert isinstance(cell_color_dict, dict), "cell_color_dict should be a dict"

    def test_rename_milestone(self):
        mw = self.milestone_wrapper
        milestone_old2new = {
            "W": "A",
            "X": "B",
            "Y": "C",
            "Z": "D",
            "A": "E",
        }
        mw.rename_milestone(milestone_old2new)
        assert set(mw.id_list) == set(milestone_old2new.values())

    def test_subset_by_cells(self):
        mw = self.milestone_wrapper
        cell_id_list = ["b", "c", "e"]
        new_mw = mw.subset_by_cells(cell_id_list=cell_id_list)
        assert set(new_mw.cell_id_list) == set(cell_id_list)

    def test_subset_by_edges(self):
        mw = self.milestone_wrapper
        edge_list = [("X", "Z"), ("Z", "A")]
        new_mw = mw.subset_by_edges(edge_list=edge_list)
        assert set([tuple(i) for i in new_mw.milestone_network[["from", "to"]].values.tolist()]) == set(edge_list)

    def test_remove_loop_edge(self):
        mw = self.milestone_wrapper
        mn = mw.milestone_network
        # add new loop row to milestone network
        loop_row = pd.Series(["W", "W", 1, True], index=mn.columns)
        new_mn = mn.append(loop_row, ignore_index=True)
        mw.milestone_network = new_mn
        # excute remove loop
        mw.remove_loop_edges()

        assert compare_dataframes(mn, mw.milestone_network, on_columns=["from", "to"])

    def test_is_connected(self):
        mw = self.milestone_wrapper
        assert mw.is_connected(), "milestone network should be connected"

    def test_get_root_milestone(self):
        mw = self.milestone_wrapper
        root_milestone = mw.get_root_milestone()
        assert root_milestone == "W", f"root milestone should be 'W', but got {root_milestone}"

    def test_merge_edge_trajectory(self):
        pass

    def test_merge_milestone_trajectory(self):
        test_data = get_merge_milestone_trajectory_mw()
        global_milestone_wrapper = test_data["global_milestone_wrapper"]
        local_milestone_wrapper = test_data["local_milestone_wrapper"]
        replace_milestone = test_data["replace_milestone"]
        expected_merged_mw = test_data["merged_milestone_wrapper"]

        merged_mw = global_milestone_wrapper.merge_milestone_trajectory(
            sub_mw=local_milestone_wrapper,
            replace_milestone=replace_milestone,
        )

        assert compare_dataframes(merged_mw.milestone_network, expected_merged_mw.milestone_network, on_columns=["from", "to"])
        assert compare_dataframes(merged_mw.progressions, expected_merged_mw.progressions, on_columns=["from", "to"])
        assert compare_dataframes(merged_mw.divergence_regions, expected_merged_mw.divergence_regions, on_columns=["divergence_id", "milestone_id"])
        # assert merged_mw.progressions
        # assert merged_mw.divergence_regions
        # merged_prog = merged_mw.progressions


if __name__ == "__main__":
    pytest.main(["-v", __file__])

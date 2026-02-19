import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy.sparse import csc_matrix

import cafe

from ..test_util import compare_dataframes, compare_dataframes_closely


def setup_method_data():
    counts = np.array(
        [
            [0, 10],
            [8, 10],
            [12, 12],
            [20, 20],
            [15, 16],
            [22, 20],
        ]
    )

    counts = csc_matrix(counts)

    fadata = cafe.data.FateAnnData(X=counts)
    fadata.obs.index = ["a", "b", "c", "d", "e", "f"]
    fadata.obs["clusters"] = [1, 1, 2, 2, 2, 3]
    fadata.obs["clusters"] = fadata.obs["clusters"].astype("category")
    fadata.var.index = ["g1", "g2"]
    fadata.layers["counts"] = counts
    fadata.layers["expression"] = counts.copy()
    fadata.obsm["X_emb"] = counts.toarray().copy()

    return fadata


class TestFateAnnData:
    def setup_method(self):
        self.fadata = setup_method_data()

    def test_init(self):
        assert isinstance(self.fadata, ad.AnnData)
        assert self.fadata.shape == (6, 2)
        assert "cafe" in self.fadata.uns.keys()

    def test_from_anndata(self):
        # data source: https://github.com/theislab/cellrank_reproducibility/blob/master/data/dyngen_simulated_data/bifurcating.h5ad
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/bifurcating.h5ad")
        fadata = cafe.data.FateAnnData.from_anndata(adata)
        assert fadata.id is not None

    def test_to_anndata(self):
        fadata = self.fadata
        adata = fadata.to_anndata(delete_trajectory=True)
        assert isinstance(adata, ad.AnnData)

    @pytest.mark.skipif(not cafe.settings.r_available, reason="R is not available")
    def test_read_dynverse_simulation_data(self):
        fadata = cafe.data.FateAnnData.read_dynverse_simulation_data()
        assert fadata.is_wrapped_with_trajectory

    def test_add_model_name(self):
        # test in test_read_dynverse_simulation_data
        pass

    def test_get_all_model_name(self):
        # first model
        fadata = self.fadata
        fadata.add_model_name("first model")
        self.test_add_trajectory()
        # second model
        milestone_wrapper = fadata.milestone_wrapper
        from cafe.util import random_time_string

        # radom_time_string for parsing
        fadata.add_model_name(random_time_string("second model"))
        fadata.add_trajectory(
            milestone_network=milestone_wrapper.milestone_network,
            divergence_regions=milestone_wrapper.divergence_regions,
            milestone_percentages=milestone_wrapper.milestone_percentages,
        )

        model_name_list = self.fadata.get_all_model_name()
        assert sorted(model_name_list) == sorted(["first model", "second model"])

    def test_subset_trajectory(self):
        self.test_add_trajectory()
        edge_list = [("X", "Z"), ("Z", "A")]
        fadata_subset = self.fadata.subset_trajectory(edge_list=edge_list)
        mw = fadata_subset.milestone_wrapper
        assert set([tuple(i) for i in mw.milestone_network[["from", "to"]].values.tolist()]) == set(edge_list)

    def test_get_item(self):
        # test obs index
        self.test_add_trajectory()
        self.fadata[:3]  # first 3 cells

    def test_get_copy(self):
        self.test_add_trajectory()
        self.fadata.copy()

    def test_add_prior_information(self):
        fadata = self.fadata
        start_cell = "a"
        fadata.add_prior_information(start_cell=start_cell)
        assert fadata.prior_information["start_cell"] == start_cell

        self.fadata.add_prior_information(start_id="a", group_id=self.fadata.obs["clusters"].tolist())
        self.fadata.add_prior_information(end_id="f")
        assert set(["start_id", "group_id", "end_id"]) <= set(self.fadata.prior_information.keys())

    def test_recognize_prior_information(self):
        fadata = self.fadata
        fadata.recognize_prior_information()
        assert fadata.prior_information["cluster"] == "clusters"
        assert fadata.prior_information["basis"] == "X_emb"

    def test_add_trajectory(self):
        from .test_fate_milestone_wrapper import setup_method_data

        milestone_wrapper = setup_method_data()
        self.fadata.add_trajectory(
            milestone_network=milestone_wrapper.milestone_network,
            divergence_regions=milestone_wrapper.divergence_regions,
            milestone_percentages=milestone_wrapper.milestone_percentages,
        )
        assert self.fadata.is_wrapped_with_trajectory

    def test_add_trajectory_mannually(self):
        # mannully specific milestone edge
        milestone_network = pd.DataFrame(
            data=[
                [1, 2],
                [2, 3],
            ],
            columns=["from", "to"],
        )
        self.fadata.add_trajectory_mannually(milestone_network=milestone_network, cluster_key="clusters", basis="X_emb")

        assert self.fadata.is_wrapped_with_trajectory

    def test_add_waypoints(self):
        # from .test_fate_milestone_wrapper import setup_method_data
        # milestone_wrapper = setup_method_data()
        self.test_add_trajectory()
        self.fadata.add_waypoints()
        assert self.fadata.is_wrapped_with_waypoints
        # TODO：test write_h5ad
        # self.fadata.write_h5ad("test_fate_anndata.h5ad")
        # fadata = cafe.data.read_h5ad("test_fate_anndata.h5ad")
        # assert fadata.waypoint_wrapper is not None

    def test_get_start_milestone(self):
        self.test_add_trajectory_mannually()
        start_milestone = self.fadata.get_start_milestone("a")
        assert start_milestone == 1

    def test_get_trajectory_pseudotime_by_milestone(self):
        self.test_add_trajectory_mannually()
        pseudotime = self.fadata.get_trajectory_pseudotime(start_milestone=1)
        assert len(pseudotime) == self.fadata.shape[0]  # assert pseudotime length is equal to cell num

    def test_get_trajectory_pseudotime_by_cell(self):
        self.test_add_trajectory_mannually()
        self.fadata.add_prior_information(start_cell="a")
        pseudotime = self.fadata.get_trajectory_pseudotime()  # extract start_cell from prior information
        assert len(pseudotime) == self.fadata.shape[0]

    def test_get_trajectory_pseudo_velocity(self):
        # divergence test case is in ../metric/test_metric_velocity.py
        self.test_add_trajectory_mannually()
        self.fadata.add_prior_information(basis="X_emb")
        pseudo_velocity = self.fadata.get_trajectory_pseudo_velocity()  # extract start_cell from prior information
        assert pseudo_velocity.shape == self.fadata.obsm["X_emb"].shape

    def test_write(self):
        self.test_add_waypoints()
        self.fadata.write_h5ad(f"{os.path.dirname(__file__)}/bifurcating_fadata.h5ad")

    def test_add_trajectory_branch(self):
        # input data
        branch_network = pd.DataFrame(
            columns=["from", "to"],
            data=[
                ["A", "B"],
                ["A", "C"],
                ["B", "D"],
            ],
        )
        branch_progressions = pd.DataFrame(
            columns=["cell_id", "branch_id", "percentage"],
            data=[
                ["a", "A", 0.0],
                ["b", "A", 0.8],
                ["c", "B", 0.2],
                ["d", "B", 1.0],
                # compared to "test_add_trajectory" test case, cell "e" is moved to branch "C" from divergence region
                ["e", "C", 0.2],
                ["f", "D", 0.2],
            ],
        )
        branches = pd.DataFrame(
            columns=["branch_id", "length", "directed"],
            data=[
                ["A", 1.0, True],
                ["B", 1.0, True],
                ["C", 1.0, True],
                ["D", 2.0, True],
            ],
        )

        # execute function
        self.fadata.add_trajectory_branch(branch_network, branch_progressions, branches)

        # expected result
        expected_milestone_network = pd.DataFrame(
            columns=["from", "to", "length", "directed"],
            data=[
                ["1", "2", 1.0, True],
                ["2", "3", 1.0, True],
                ["2", "4", 1.0, True],
                ["3", "5", 2.0, True],
            ],
        )
        expected_progressions = pd.DataFrame(
            columns=["cell_id", "from", "to", "percentage"],
            data=[
                ["a", "1", "2", 0.0],
                ["b", "1", "2", 0.8],
                ["c", "2", "3", 0.2],
                ["d", "2", "3", 1.0],
                ["e", "2", "4", 0.2],
                ["f", "3", "5", 0.2],
            ],
        )

        # assert
        milestone_wrapper = self.fadata.milestone_wrapper
        assert compare_dataframes(milestone_wrapper["milestone_network"], expected_milestone_network, on_columns=["from", "to"])
        assert compare_dataframes(milestone_wrapper["progressions"], expected_progressions, on_columns=["cell_id", "from", "to"])

    def get_add_trajectory_linear_test_data(self):
        # new test case: pseudotime and FateAnnData
        name = "test_add_trajectory_linear"
        cell_ids = ["a", "b", "c", "d", "e", "f"]
        pseudotime = [0.0, 0.1, 0.4, 0.5, 0.8, 1.0]

        expression = np.tile(pseudotime, (2, 1)).T
        fadata = cafe.data.FateAnnData(X=expression, name=name)
        fadata.obs.index = cell_ids
        fadata.layers["expression"] = expression.copy()

        test_data = {
            "fadata": fadata,
            "pseudotime": pseudotime,
            "cell_ids": cell_ids,
        }
        return test_data

    def test_add_trajectory_linear(self):
        # input data
        test_data = self.get_add_trajectory_linear_test_data()
        fadata = test_data["fadata"]
        pseudotime = test_data["pseudotime"]
        cell_ids = test_data["cell_ids"]

        # execute function
        fadata.add_trajectory_linear(pseudotime)

        # expected result
        expected_milestone_ids = ["milestone_begin", "milestone_end"]
        expected_milestone_network = pd.DataFrame(
            {
                "from": "milestone_begin",
                "to": "milestone_end",
                "length": 1,
                "directed": True,
            },
            index=[0],
        )
        expected_progressions = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "from": "milestone_begin",
                "to": "milestone_end",
                "percentage": pseudotime,
            }
        )

        # assert
        assert fadata.milestone_wrapper["id_list"] == expected_milestone_ids
        assert fadata.milestone_wrapper["milestone_network"].equals(expected_milestone_network)
        assert fadata.milestone_wrapper["progressions"].equals(expected_progressions)

    def test_add_trajectory_cycle(self):
        # input data
        test_data = self.get_add_trajectory_linear_test_data()
        fadata = test_data["fadata"]
        pseudotime = test_data["pseudotime"]
        # cell_ids = test_data["cell_ids"]

        # execute function
        fadata.add_trajectory_cycle(pseudotime)

        # expected result
        expected_milestone_ids = ["A", "B", "C"]
        expected_milestone_network = pd.DataFrame(
            columns=[
                "from",
                "to",
                "length",
                "directed",
            ],
            data=[["A", "B", 1, False], ["B", "C", 1, False], ["C", "A", 1, False]],
        )
        expected_progressions = pd.DataFrame(
            columns=["cell_id", "from", "to", "percentage"],
            data=[
                ["a", "A", "B", 0],
                ["b", "A", "B", 0.3],
                ["c", "B", "C", 0.2],
                ["d", "B", "C", 0.5],
                ["e", "C", "A", 0.4],
                ["f", "C", "A", 1],
            ],
        )

        # assert
        milestone_wrapper = fadata.milestone_wrapper
        assert milestone_wrapper["id_list"] == expected_milestone_ids
        assert compare_dataframes_closely(milestone_wrapper["milestone_network"], expected_milestone_network, on_columns=["from", "to"])
        assert compare_dataframes_closely(milestone_wrapper["progressions"], expected_progressions, on_columns=["cell_id"])

    def get_add_trajectory_probability_test_data(self):
        id = "test_add_end_state_probabilities"
        cell_ids = ["a", "aa", "b", "bb", "c", "cc"]
        fdata = cafe.data.FateAnnData(X=np.zeros((len(cell_ids), 2)), name=id)
        end_state_ids = ["A", "B", "C"]
        end_state_probabilities = pd.DataFrame(
            columns=["cell_id", "A", "B", "C"],
            data=[
                ["a", 0.5, 0, 0],
                ["aa", 1, 0, 0],
                ["b", 0, 0.5, 0],
                ["bb", 0, 1, 0],
                ["c", 0, 0, 0.5],
                ["cc", 0, 0, 1],
            ],
        )
        pseudotime = [0.5, 1, 0.5, 1, 0.5, 1]
        pseudotime = pd.Series(pseudotime, index=cell_ids)
        test_data = {
            "id": id,
            "cell_ids": cell_ids,
            "fadata": fdata,
            "end_state_ids": end_state_ids,
            "end_state_probabilities": end_state_probabilities,
            "pseudotime": pseudotime,
        }
        return test_data

    def test_add_trajectory_probablity_3_states(self):
        # input data
        test_data = self.get_add_trajectory_probability_test_data()
        fadata = test_data["fadata"]
        end_state_probabilities = test_data["end_state_probabilities"]
        end_state_ids = test_data["end_state_ids"]
        pseudotime = test_data["pseudotime"]

        # execute function
        fadata.add_trajectory_probability(
            end_state_probabilities=end_state_probabilities,
            pseudotime=pseudotime,
        )

        # expected result
        start_milestone_id = "milestone_begin"
        milestone_ids = [start_milestone_id] + end_state_ids
        expected_milestone_network = pd.DataFrame({"from": start_milestone_id, "to": end_state_ids, "length": 1, "directed": True})
        expected_divergence_regions = pd.DataFrame(
            {
                "milestone_id": milestone_ids,
                "divergence_id": "D",
                "is_start": pd.Series(milestone_ids) == start_milestone_id,
            }
        )
        scaled_pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
        expected_progressions = end_state_probabilities.melt(id_vars=["cell_id"], var_name="to", value_name="percentage")
        expected_progressions["from"] = start_milestone_id
        expected_progressions["percentage"] = expected_progressions.groupby("cell_id")["percentage"].transform(
            lambda x: x / x.sum() * scaled_pseudotime[x.name]
        )  # 缩放使其之和为1，暂时不理解这个
        expected_progressions = expected_progressions[["cell_id", "from", "to", "percentage"]]

        # assert
        milestone_wrapper = fadata.milestone_wrapper
        assert milestone_wrapper["milestone_network"].equals(expected_milestone_network)
        assert milestone_wrapper["divergence_regions"].equals(expected_divergence_regions)
        assert milestone_wrapper["progressions"].equals(expected_progressions)

    def test_add_trajectory_probablity(self):
        # input data
        test_data = self.get_add_trajectory_probability_test_data()
        fadata = test_data["fadata"]
        end_state_probabilities = test_data["end_state_probabilities"]
        pseudotime = test_data["pseudotime"]
        # no terminal state
        end_state_probabilities = end_state_probabilities["cell_id"].to_frame()

        # execute function
        fadata.add_trajectory_probability(
            end_state_probabilities=end_state_probabilities,
            pseudotime=pseudotime,
        )
        milestone_wrapper = fadata.milestone_wrapper

        # expected result
        fadata.add_trajectory_linear(
            pseudotime=pseudotime,
            directed=True,
        )
        excepted_milestone_network = fadata.milestone_wrapper

        # assert
        assert milestone_wrapper["milestone_network"].equals(excepted_milestone_network["milestone_network"])
        assert milestone_wrapper["divergence_regions"].equals(excepted_milestone_network["divergence_regions"])
        assert milestone_wrapper["progressions"].equals(excepted_milestone_network["progressions"])

    def test_add_trajectory_cluster(self):
        # input data
        from .test_fate_milestone_wrapper import setup_method_data

        milestone_wrapper = setup_method_data()
        fadata = self.fadata
        milestone_network = milestone_wrapper["milestone_network"]
        cluster_list = ["W", "X", "X", "Z", "Z", "Z"]

        # execute function
        fadata.add_trajectory_cluster(
            milestone_network=milestone_network,
            cluster=cluster_list,
        )

        # assert
        assert fadata.milestone_wrapper["milestone_percentages"].query("`percentage`==1")["milestone_id"].tolist() == cluster_list

    def get_add_trajectory_projection_test_data(self):
        self.test_add_trajectory()
        fadata = self.fadata
        milestone_wrapper = fadata.milestone_wrapper
        milestone_network = milestone_wrapper["milestone_network"]
        X_emb = pd.DataFrame(
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
        milestone_emb = pd.DataFrame(
            columns=["milestone_id", "comp_1", "comp_2"],
            data=[
                ["W", 0, 1],
                ["X", 1, 1],
                ["Y", 1, 2],
                ["Z", 2, 1],
                ["A", 4, 1],
            ],
        )
        X_emb.set_index("cell_id", inplace=True)
        milestone_emb.set_index("milestone_id", inplace=True)

        # expected result
        expected_progressions = pd.DataFrame(
            columns=["cell_id", "from", "to", "percentage"],
            data=[
                ["a", "W", "X", 0],
                ["b", "W", "X", 0.8],
                ["c", "X", "Z", 0.2],
                ["d", "X", "Z", 1],
                ["e", "X", "Y", 0.5],
                ["f", "Z", "A", 0.2],
            ],
        )

        test_data = {
            "fadata": fadata,
            "X_emb": X_emb,
            "milestone_emb": milestone_emb,
            "milestone_network": milestone_network,
            "expected_progressions": expected_progressions,
        }

        return test_data

    def test_add_trajectory_projection(self):
        # input data
        test_data = self.get_add_trajectory_projection_test_data()
        fadata = test_data["fadata"]
        X_emb = test_data["X_emb"]
        milestone_emb = test_data["milestone_emb"]
        milestone_network = test_data["milestone_network"]

        # execute function
        fadata.add_trajectory_projection(
            milestone_network=milestone_network,
            milestone_emb=milestone_emb,
            X_emb=X_emb,
        )

        # expected result
        expected_progressions = test_data["expected_progressions"]

        # assert
        assert compare_dataframes_closely(fadata.milestone_wrapper["progressions"], expected_progressions, on_columns="cell_id")

    def test_add_trajectory_projection_with_cluster(self):
        # input data
        test_data = self.get_add_trajectory_projection_test_data()
        fadata = test_data["fadata"]
        X_emb = test_data["X_emb"]
        milestone_emb = test_data["milestone_emb"]
        milestone_network = test_data["milestone_network"]

        # execute function
        cluster_key = "clusters"
        fadata.obs[cluster_key] = [
            "X",
            "X",
            "X",
            "Z",
            "Z",
            "Z",
        ]  # add cluster, cluster names should be consistent with milestone names
        fadata.add_trajectory_projection(
            milestone_network=milestone_network,
            milestone_emb=milestone_emb,
            X_emb=X_emb,
            cluster_key=cluster_key,
        )

        # expected result
        expected_progressions = pd.DataFrame(
            columns=["cell_id", "from", "to", "percentage"],
            data=[
                ["a", "W", "X", 0],
                ["b", "W", "X", 0.8],
                ["c", "X", "Z", 0.2],
                ["d", "X", "Z", 1],
                ["e", "X", "Z", 0.2],  # e -> X-Z edge
                ["f", "Z", "A", 0.2],
            ],
        )

        # assert
        assert compare_dataframes_closely(fadata.milestone_wrapper["progressions"], expected_progressions, on_columns="cell_id")

    def test_add_trajectory_graph(self):
        # input data
        name = "test_add_trajectory_cell_graph"
        cell_ids = ["W", "X", "Y", "Z", "A", "WbX", "XcZ", "XeY", "ZfA", "a", "b", "c", "d", "e", "f"]
        expression = np.zeros([len(cell_ids), 2])
        fadata = cafe.data.FateAnnData(X=expression, name=name)
        fadata.obs.index = cell_ids

        cell_graph = pd.DataFrame(
            columns=["from", "to", "length", "directed"],
            data=[
                ["W", "WbX", 0.8, False],
                ["WbX", "X", 0.2, False],
                ["X", "XeY", 0.5, False],
                ["XeY", "Y", 0.5, False],
                ["X", "XcZ", 0.2, False],
                ["XcZ", "Z", 0.8, False],
                ["Z", "ZfA", 0.2, False],
                ["ZfA", "A", 0.8, False],
                ["W", "a", 0.5, False],
                ["WbX", "b", 0.5, False],
                ["XcZ", "c", 0.5, False],
                ["Z", "d", 0.5, False],
                ["XeY", "e", 0.2, False],
                ["ZfA", "f", 0.5, False],
            ],
        )
        cell_graph["directed"] = True  # easier for directed graph

        to_keep = dict(
            W=True,
            X=True,
            Y=True,
            Z=True,
            A=True,
            WbX=True,
            XcZ=True,
            XeY=True,
            ZfA=True,
            a=False,
            b=False,
            c=False,
            d=False,
            e=False,
            f=False,
        )
        to_keep = pd.Series(to_keep)

        # execute function
        fadata.add_trajectory_graph(
            cell_graph=cell_graph,
            to_keep=to_keep,
            milestone_prefix="ML_",
        )

        # expected result
        expected_milestone_ids = [f"ML_{i}" for i in ["W", "X", "Y", "A"]]
        expected_milestone_network = pd.DataFrame(
            columns=["from", "to", "length", "directed"],
            data=[
                ["ML_W", "ML_X", 1, False],
                ["ML_X", "ML_Y", 1, False],
                ["ML_X", "ML_A", 2, False],
            ],
        )
        # easier for directed graph
        expected_milestone_network["directed"] = True

        expected_progressions = pd.DataFrame(
            columns=["cell_id", "from", "to", "percentage"],
            data=[
                ["W", "ML_W", "ML_X", 0],
                ["X", "ML_W", "ML_X", 1],
                ["Y", "ML_X", "ML_Y", 1],
                ["Z", "ML_X", "ML_A", 0.5],
                ["A", "ML_X", "ML_A", 1],
                ["WbX", "ML_W", "ML_X", 0.8],
                ["XcZ", "ML_X", "ML_A", 0.1],
                ["XeY", "ML_X", "ML_Y", 0.5],
                ["ZfA", "ML_X", "ML_A", 0.6],
                ["a", "ML_W", "ML_X", 0],
                ["b", "ML_W", "ML_X", 0.8],
                ["c", "ML_X", "ML_A", 0.1],
                ["d", "ML_X", "ML_A", 0.5],
                ["e", "ML_X", "ML_Y", 0.5],
                ["f", "ML_X", "ML_A", 0.6],
            ],
        )

        # assert
        milestone_wrapper = fadata.milestone_wrapper
        assert sorted(milestone_wrapper["id_list"]) == sorted(expected_milestone_ids)
        assert compare_dataframes_closely(milestone_wrapper["milestone_network"], expected_milestone_network, on_columns=["from", "to"])
        assert compare_dataframes_closely(milestone_wrapper["progressions"], expected_progressions, on_columns=["cell_id"])

    def test_add_trajectory_lineage(self):
        name = "test_add_trajectory_lineage"
        cell_ids = ["a1", "a2", "b1", "b2", "c1", "c2", "c3", "d1", "d2", "d3"]
        expression = np.zeros([len(cell_ids), 2])
        fadata = cafe.data.FateAnnData(X=expression, name=name)
        fadata.obs.index = cell_ids

        cluster_key = "clusters"
        fadata.obs[cluster_key] = ["A", "A", "B", "B", "C", "C", "C", "D", "D", "D"]
        probability = pd.DataFrame(
            columns=["C", "D"],
            index=cell_ids,
            data=[
                [0, 0.1],
                [0.2, 0.3],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.6, 0.6],
                [0.8, 0.6],
                [1, 0.3],
                [0.6, 0.7],
                [0.6, 0.8],
                [0.3, 0.9],
            ],
        )

        # execute function
        fadata.add_trajectory_lineage(probability=probability, cluster_key=cluster_key)

        # expected result
        # expected_milestone_ids = ["A", "B", "C", "D"]
        expected_milestone_network = pd.DataFrame(
            columns=["from", "to", "length", "directed"],
            data=[
                ["A", "B", 1, True],
                ["B", "C", 1, True],
                ["B", "D", 1, True],
            ],
        )
        # expected_progressions = pd.DataFrame(
        #     columns=["cell_id", "from", "to", "percentage"],
        #     data=[
        #         ["a1", "A", "B", 0],
        #         ["a2", "A", "B", 1 / 3],
        #         ["b1", "A", "B", 2 / 3],
        #         ["b2", "B", "C", 0],
        #         ["b2", "B", "D", 0],
        #         ["c1", "B", "C", 0],
        #         ["c1", "B", "D", 0],
        #         ["c2", "B", "C", 0],
        #         ["c2", "B", "D", 0],
        #         ["c3", "B", "C", 0],
        #         ["d1", "B", "C", 0],
        #         ["d1", "B", "D", 0],
        #         ["d2", "B", "C", 0],
        #         ["d2", "B", "D", 0],
        #         ["d3", "B", "D", 0],
        #     ],
        # )
        expected_divergence_regions = pd.DataFrame(
            columns=["milestone_id", "divergence_id", "is_start"],
            data=[
                ["B", "BCD", True],
                ["C", "BCD", False],
                ["D", "BCD", False],
            ],
        )

        # TODO: assert
        milestone_wrapper = fadata.milestone_wrapper
        assert expected_milestone_network.equals(milestone_wrapper["milestone_network"])
        assert expected_divergence_regions.equals(milestone_wrapper["divergence_regions"])

    @pytest.mark.skip("velocity_graph is need for add velocity trajectory")
    def test_add_trajectory_velocity(self):
        # TODO: paga reference
        name = "test_add_trajectory_velocity"
        # cell_ids = ["a1", "b1", "b2", "c1", "d1"]
        cluster_key = "clusters"
        cluster_list = ["a", "b", "b", "c", "d"]
        X_emb = np.array(
            [
                [0, 1],
                [1, 2],
                [1, 0],
                [2, 2],
                [2, 0],
            ]
        )
        fadata = cafe.data.FateAnnData(X=X_emb, name=name)
        fadata.obs[cluster_key] = cluster_list
        fadata.obsm["X_umap"] = X_emb
        fadata.layers["spliced"] = X_emb
        fadata.layers["unspliced"] = X_emb
        velocity = np.array(
            [
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
            ]
        )

        # # mannual neighbors
        # from scipy.sparse import csr_matrix
        # distances = np.array([
        #         [0, 1.414, 1.414, 0, 0],
        #         [1.414, 0, 0, 1, 0],
        #         [1.414, 0, 0, 0, 1],
        #         [0,1,0,0,0],
        #         [0,0,1,0,0],
        #         ])
        # distances = csr_matrix(distances)
        # connectivities = np.array([
        #         [0,1,1,0,0],
        #         [1,0,0,1,0],
        #         [1,0,0,0,1],
        #         [0,1,0,0,0],
        #         [0,0,1,0,0]
        #         ])
        # connectivities = csr_matrix(connectivities)
        # neighbors = {
        #     "distances": distances,
        #     "connectivities": connectivities,
        # }
        # automatic neighbors, don't meet the demand
        sc.pp.neighbors(fadata, n_neighbors=3)
        neighbors = {"distances": fadata.obsp["distances"], "connectivities": fadata.obsp["connectivities"]}
        n_obs = fadata.shape[0]
        # TODO: velocity_graph is need for add velocity trajectory
        velocity_graph = np.random.rand(n_obs, n_obs)
        velocity_graph_neg = np.random.rand(n_obs, n_obs)

        # expected_milestone_network = pd.DataFrame(
        #     columns=["from", "to", "length", "directed"],
        #     data=[
        #         ["a", "b", 1, True],
        #         ["b", "c", 1, True],
        #         ["b", "d", 1, True],
        #     ],
        # )

        fadata.add_trajectory_velocity(
            velocity=velocity,
            velocity_graph=velocity_graph,
            velocity_graph_neg=velocity_graph_neg,
            neighbors=neighbors,
            cluster_key=cluster_key,
        )

        # milestone_wrapper = fadata.milestone_wrapper
        # PAGA result can't be expected.
        # assert expected_milestone_network.equals(milestone_wrapper["milestone_network"])

    @pytest.mark.skip("velocity_graph is need for add velocity trajectory")
    def test_add_trajectory_velocity2(self):
        name = "test_add_trajectory_velocity2"
        # cell_ids = ["a1", "b1", "b2", "c1", "d1"]
        cluster_key = "clusters"
        cluster_list = ["a", "b", "b", "c", "d"]
        X_emb = np.array(
            [
                [0, 1],
                [1, 2],
                [1, 0],
                [2, 2],
                [2, 0],
            ]
        )
        fadata = cafe.data.FateAnnData(X=X_emb, name=name)
        fadata.obs[cluster_key] = cluster_list
        fadata.obsm["X_umap"] = X_emb
        fadata.layers["spliced"] = X_emb
        fadata.layers["unspliced"] = X_emb
        velocity = np.array(
            [
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
            ]
        )

        # 上下左右抖动0.5
        fadata_list = [fadata]
        for move in [[0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5]]:
            tmp_fadata = fadata.copy()
            tmp_X_emb = X_emb + move
            tmp_fadata.X = tmp_X_emb
            tmp_fadata.obsm["X_umap"] = tmp_X_emb
            tmp_fadata.layers["spliced"] = tmp_X_emb
            tmp_fadata.layers["unspliced"] = tmp_X_emb
            fadata_list.append(tmp_fadata)
        velocity = np.repeat([[1, 0]], 25, axis=0).reshape(25, 2)
        fadata = cafe.data.FateAnnData.from_anndata(sc.concat(fadata_list))
        fadata.obs.index = range(fadata.shape[0])
        print(fadata)

        # automatic neighbors, don't meet the demand
        sc.pp.neighbors(fadata, n_neighbors=3)
        neighbors = {"distances": fadata.obsp["distances"], "connectivities": fadata.obsp["connectivities"]}

        # expected_milestone_network = pd.DataFrame(
        #     columns=["from", "to", "length", "directed"],
        #     data=[
        #         ["a", "b", 1, True],
        #         ["b", "c", 1, True],
        #         ["b", "d", 1, True],
        #     ],
        # )

        fadata.add_trajectory_velocity(
            velocity=velocity,
            neighbors=neighbors,
            cluster_key=cluster_key,
        )

        # milestone_wrapper = fadata.milestone_wrapper
        # PAGA result can't be expected.
        # assert expected_milestone_network.equals(milestone_wrapper["milestone_network"])

    def test_group_onto_trajectory_edges(self):
        # input data
        self.test_add_trajectory()  # reuse test case from test_add_trajectory
        fadata = self.fadata
        cluster_key = "group"

        # execute function
        fadata.group_onto_trajectory_edges(cluster_key=cluster_key)

        # expected result
        excepted_group = ["W->W", "W->X", "X->Z", "Z->Z", "X->Z", "Z->A"]

        # assert
        assert cluster_key in self.fadata.obs.columns
        assert excepted_group == self.fadata.obs[cluster_key].tolist()

    def test_group_onto_nearest_milestones(self):
        # input data
        self.test_add_trajectory()  # reuse test case from test_add_trajectory
        fadata = self.fadata
        cluster_key = "group"

        # execute function
        fadata.group_onto_nearest_milestones(cluster_key=cluster_key)

        # expected result
        excepted_group = ["W", "X", "X", "Z", "Z", "Z"]

        # assert
        assert cluster_key in self.fadata.obs.columns
        assert excepted_group == self.fadata.obs[cluster_key].tolist()

    def get_simplify_trajectory_test_data_linear(self):
        # input data
        id = "linear_directed"
        cell_ids = ["a", "b", "c", "d", "e"]
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
        fadata = cafe.data.FateAnnData(name=id, X=np.zeros((len(cell_ids), 2)))
        fadata.add_trajectory(milestone_network=milestone_network, progressions=progressions)

        # expected result
        expected_milestone_network = pd.DataFrame(
            data=[["A", "D", 3, True]],
            columns=["from", "to", "length", "directed"],
        )
        expected_progressions = pd.DataFrame(
            data=[
                ["a", "A", "D", 0.1],
                ["b", "A", "D", 0.2],
                ["c", "A", "D", 0.4],
                ["d", "A", "D", 0.6],
                ["e", "A", "D", 0.8],
            ],
            columns=["cell_id", "from", "to", "percentage"],
        )

        test_data = {
            "id": id,
            "cell_ids": cell_ids,
            "milestone_network": milestone_network,
            "progressions": progressions,
            "fadata": fadata,
            "expected_milestone_network": expected_milestone_network,
            "expected_progressions": expected_progressions,
        }
        return test_data

    def test_simplify_trajectory_linear_directed(self):
        # input data
        test_data = self.get_simplify_trajectory_test_data_linear()
        fadata = test_data["fadata"]

        # execute function
        simplified_milestone_wrapper = fadata.simplify_trajectory()

        # expected result
        expected_milestone_network = test_data["expected_milestone_network"]
        expected_progressions = test_data["expected_progressions"]

        # assert
        assert simplified_milestone_wrapper.milestone_network.equals(expected_milestone_network)
        assert compare_dataframes_closely(simplified_milestone_wrapper.progressions, expected_progressions, on_columns="cell_id")

    def test_simplify_trajectory_linear_undirected(self):
        # input data
        test_data = self.get_simplify_trajectory_test_data_linear()
        id = "linear_undirected"
        cell_ids = test_data["cell_ids"]
        milestone_network = test_data["milestone_network"]
        progressions = test_data["progressions"]
        fadata = cafe.data.FateAnnData(name=id, X=np.zeros((len(cell_ids), 2)))
        milestone_network["directed"] = False  # undirected graph
        fadata.add_trajectory(milestone_network=milestone_network, progressions=progressions)

        # execute function
        simplified_milestone_wrapper = fadata.simplify_trajectory()

        # expected result
        expected_milestone_network = test_data["expected_milestone_network"]
        expected_milestone_network["directed"] = False
        expected_progressions = test_data["expected_progressions"]

        # assert
        assert simplified_milestone_wrapper.milestone_network.equals(expected_milestone_network)
        assert compare_dataframes_closely(simplified_milestone_wrapper.progressions, expected_progressions, on_columns="cell_id")

    def get_simplify_trajectory_test_data_bifurcation(self):
        # input data
        id = "bifurcation_directed"
        cell_ids = ["a", "b", "c", "d", "e", "f"]
        # milestone_ids = ["A", "B", "C", "D", "E", "F", "G"]
        milestone_network = pd.DataFrame(
            data=[
                ["A", "B", 4, True],
                ["A", "C", 4, True],
                ["B", "D", 1, True],
                ["C", "E", 1, True],
                ["E", "F", 1, True],
                ["E", "G", 1, True],
            ],
            columns=["from", "to", "length", "directed"],
        )
        progressions = pd.DataFrame(
            data=[
                ["a", "A", "B", 0.5],
                ["b", "A", "C", 0.5],
                ["c", "B", "D", 0.5],
                ["d", "C", "E", 0.5],
                ["e", "E", "F", 0.5],
                ["f", "E", "G", 0.5],
            ],
            columns=["cell_id", "from", "to", "percentage"],
        )

        fadata = cafe.data.FateAnnData(name=id, X=np.zeros((len(cell_ids), 2)))
        fadata.add_trajectory(milestone_network=milestone_network, progressions=progressions)

        # expected result
        expected_milestone_network = pd.DataFrame(
            data=[["A", "D", 5, True], ["A", "E", 5, True], ["E", "F", 1, True], ["E", "G", 1, True]],
            columns=["from", "to", "length", "directed"],
        )
        expected_progressions = pd.DataFrame(
            data=[
                ["a", "A", "D", 0.4],
                ["b", "A", "E", 0.4],
                ["c", "A", "D", 0.9],
                ["d", "A", "E", 0.9],
                ["e", "E", "F", 0.5],
                ["f", "E", "G", 0.5],
            ],
            columns=["cell_id", "from", "to", "percentage"],
        )

        test_data = {
            "id": id,
            "cell_ids": cell_ids,
            "milestone_network": milestone_network,
            "progressions": progressions,
            "fadata": fadata,
            "expected_milestone_network": expected_milestone_network,
            "expected_progressions": expected_progressions,
        }
        return test_data

    def test_simplify_trajectory_bifurcation_directed(self):
        # input data
        test_data = self.get_simplify_trajectory_test_data_bifurcation()
        fadata = test_data["fadata"]

        # execute function
        simplified_milestone_wrapper = fadata.simplify_trajectory()

        # expected result
        expected_milestone_network = test_data["expected_milestone_network"]
        expected_progressions = test_data["expected_progressions"]

        # assert
        assert simplified_milestone_wrapper.milestone_network.equals(expected_milestone_network)
        assert compare_dataframes_closely(simplified_milestone_wrapper.progressions, expected_progressions, on_columns="cell_id")

    def test_simplify_trajectory_bifurcation_undirected(self):
        # input data
        test_data = self.get_simplify_trajectory_test_data_bifurcation()
        id = "bifurcation_undirected"
        cell_ids = test_data["cell_ids"]
        milestone_network = test_data["milestone_network"]
        progressions = test_data["progressions"]
        fadata = cafe.data.FateAnnData(name=id, X=np.zeros((len(cell_ids), 2)))
        milestone_network["directed"] = False  # undirected graph
        fadata.add_trajectory(milestone_network=milestone_network, progressions=progressions)

        # execute function
        simplified_milestone_wrapper = fadata.simplify_trajectory()

        # expected result
        expected_milestone_network = pd.DataFrame(
            data=[
                ["D", "E", 10, False],
                ["E", "F", 1, False],
                ["E", "G", 1, False],
            ],
            columns=["from", "to", "length", "directed"],
        )
        expected_progressions = pd.DataFrame(
            data=[
                ["a", "D", "E", 0.3],
                ["b", "D", "E", 0.7],
                ["c", "D", "E", 0.05],
                ["d", "D", "E", 0.95],
                ["e", "E", "F", 0.5],
                ["f", "E", "G", 0.5],
            ],
            columns=["cell_id", "from", "to", "percentage"],
        )

        # assert
        assert simplified_milestone_wrapper.milestone_network.equals(expected_milestone_network)
        assert compare_dataframes_closely(
            simplified_milestone_wrapper.progressions, expected_progressions, on_columns="cell_id"
        )  # TODO: 这里暂时有问题，progression里出现了milestone_network中没有的milestone


if __name__ == "__main__":
    pytest.main(["-v", __file__])

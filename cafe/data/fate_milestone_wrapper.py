# from collections.abc import Mapping
# from types import MappingProxyType
# from typing import Any

# import h5py
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .._logging import logger
from .._settings import settings
from ..util import random_time_string
from .fate_wrapper import FateWrapper

# from anndata._io.specs.registry import (  # global I/O registry
#     _REGISTRY,
#     IOSpec,
#     Reader,
#     Writer,
# )
# from anndata._types import GroupStorageType


# TODO: add docs
class MilestoneWrapper(FateWrapper):
    """Wrapper for trajectory milestones"""

    def __init__(
        self,
        milestone_network: pd.DataFrame,
        milestone_id_list: list = None,
        cell_id_list: list = None,
        divergence_regions: pd.DataFrame = None,
        milestone_percentages: pd.DataFrame = None,
        progressions: pd.DataFrame = None,
        wrapper_type: str = None,
        name="MilestoneWrapper",
        milestone_color_dict: dict = None,
    ):
        """Initialize the MilestoneWrapper class.

        Args:
            milestone_network (pd.DataFrame): milestone network with column list: ["from", "to", "length", "directed"] # TODO: confidence column for optional weak edge
            id_list(list): milstone id list, should be specified if there is a discrete milestone
            divergence_regions (pd.DataFrame, optional): divergence regions with column list: ["divergence_id", "milestone_id", "is_start"].
            milestone_percentages (pd.DataFrame, optional): milestone percentage with column list: ["cell_id", "milestone_id", "percentage"].
            progressions (pd.DataFrame, optional): progressions with  column list: ["cell_id", "from", "to", "percentage"].
            name (str, optional): name of the wrapper.

        Raises:
            ValueError: Exactly one of milestone_percentages or progressions, must be defined, the other should be None
        """
        self.id = random_time_string(name)
        self.milestone_network = self._check_milestone_network(milestone_network)
        self._milestone_network_G = None
        # if there is a discrete milestone, milestone id should be specified
        if milestone_id_list is None:
            self.id_list = milestone_network[["from", "to"]].stack().unique().tolist()
        else:
            self.id_list = milestone_id_list

        if divergence_regions is None:
            self.divergence_regions = pd.DataFrame(columns=["divergence_id", "milestone_id", "is_start"])
        else:
            self.divergence_regions = divergence_regions

        # ref: pydynverse/wrap/wrap_add_trajectory.add_trajectory
        # choose milestone_percentages or progressions
        if (milestone_percentages is None) == (progressions is None):
            if milestone_percentages is not None:
                logger.warning("Both milestone_percentages and progressions are given, will only use progressions")
                milestone_percentages = None
            else:
                raise ValueError("Exactly one of milestone_percentages or progressions, must be defined, the other should be None")
        # remove cells which are related to milestone that not shown in milestone network, （TODO: for graph mst optimization）
        # then convert to another dataframe
        if progressions is None:
            # milestone_percentages -> progressions, 'add_trajectory' test case
            milestone_percentages = MilestoneWrapper._check_milestone_percentages(milestone_network, milestone_percentages)
            progressions = MilestoneWrapper.convert_milestone_percentages_to_progressions(milestone_network, milestone_percentages)
        else:
            # progressions -> milestone_percentages, 'add_trajectory_branch' test case
            progressions = MilestoneWrapper._check_progression(milestone_network, progressions)
            milestone_percentages = MilestoneWrapper.convert_progressions_to_milestone_percentages(milestone_network, progressions)
        if cell_id_list is not None:
            self.cell_id_list = list(cell_id_list)
        elif milestone_percentages is not None:
            self.cell_id_list = milestone_percentages["cell_id"].unique().tolist()
        else:
            self.cell_id_list = progressions["cell_id"].unique().tolist()
        self.milestone_percentages = milestone_percentages
        self.progressions = progressions

        # self.classify_milestone_network()
        self.milestone_network_class = "N"
        self.directed = milestone_network["directed"].any()

        # lazy load for color
        self._milestone_color_dict = milestone_color_dict
        self._cell_color_dict = None

        self.wrapper_type = wrapper_type

    @property
    def milestone_network_G(self):
        # read
        if hasattr(self, "_milestone_network_G") and self._milestone_network_G is not None:
            pass
        else:
            self._milestone_network_G = self._convert_milestone_network_to_graph(self.milestone_network)
        return self._milestone_network_G

    @staticmethod
    def _check_milestone_percentages(milestone_network, milestone_percentages):
        valid_milestones = set(milestone_network["from"]).union(set(milestone_network["to"]))
        invalid_mask = ~milestone_percentages["milestone_id"].isin(valid_milestones)

        if invalid_mask.any():
            invalid_cells = milestone_percentages.loc[invalid_mask, "cell_id"].unique()
            logger.warning(f"dropping {len(invalid_cells)} cells because they map to milestones missing from the network.")
            milestone_percentages = milestone_percentages[~milestone_percentages["cell_id"].isin(invalid_cells)].copy()

        return milestone_percentages

    @staticmethod
    def _check_progression(milestone_network, progressions):
        valid_milestones = set(milestone_network["from"]).union(set(milestone_network["to"]))
        invalid_mask = (~progressions["from"].isin(valid_milestones)) | (~progressions["to"].isin(valid_milestones))

        if invalid_mask.any():
            invalid_cells = progressions.loc[invalid_mask, "cell_id"].unique()
            logger.warning(f"dropping {len(invalid_cells)} cells because they map to milestones missing from the network.")
            progressions = progressions[~progressions["cell_id"].isin(invalid_cells)].copy()

        return progressions

    @staticmethod
    def convert_milestone_percentages_to_progressions(milestone_network: pd.DataFrame, milestone_percentages: pd.DataFrame) -> pd.DataFrame:
        """Convert: milestone_percentages -> progressions, "add_trajectory" test case use it

        Args:
            milestone_network (pd.DataFrame): milestone network with column list: ["from", "to", "length", "directed"]
            milestone_percentages (pd.DataFrame):  milestone percentage with column list: ["cell_id", "milestone_id", "percentage"].

        Returns:
            pd.DataFrame: progressions with  column list: ["cell_id", "from", "to", "percentage"]
        """
        # part1: for cells that have 2 or more milestones
        # first merge based on "to" key result in many invalid cell_id-form relationship
        df1 = pd.merge(milestone_network, milestone_percentages, left_on="to", right_on="milestone_id")
        # second merge based on "to" key
        df2 = pd.merge(
            df1,
            milestone_percentages[["cell_id", "milestone_id"]],
            left_on=["from", "cell_id"],
            right_on=["milestone_id", "cell_id"],
        )
        # TODO: if the two step merge can be done simutaneously?
        progr_part1 = df2[["cell_id", "from", "to", "percentage"]]

        # for cells that have just 1 milestone
        # TODO: only simple reserve cells with one milestone
        progr_part2 = milestone_percentages.groupby("cell_id").filter(lambda x: len(x) == 1)
        progr_part2["from"] = progr_part2["milestone_id"]
        progr_part2["to"] = progr_part2["milestone_id"]
        progr_part2 = progr_part2[["cell_id", "from", "to", "percentage"]]

        # progressions = pd.concat([progr_part1], ignore_index=True)
        progressions = pd.concat([progr_part1, progr_part2], ignore_index=True).reset_index(drop=True)

        return progressions

    @staticmethod
    def convert_progressions_to_milestone_percentages(milestone_network: pd.DataFrame, progressions: pd.DataFrame) -> pd.DataFrame:
        """Convert: progressions -> milestone_percentages, "add_trajectory_branch" test case use it

        ref: pydynverse/wrap/convert_progressions_to_milestone_percentages.convert_progressions_to_milestone_percentages

        Args:
            milestone_network (pd.DataFrame): milestone network with column list: ["from", "to", "length", "directed"]
            progressions (pd.DataFrame): progressions with  column list: ["cell_id", "from", "to", "percentage"]

        Returns:
            pd.DataFrame: milestone percentage with column list: ["cell_id", "milestone_id", "percentage"]
        """
        # TODO: check if from milestone is the only one for each cell

        # self loops
        selfs = progressions.query("`from` == `to`")
        selfs = selfs[["cell_id", "from"]].copy().rename(columns={"from": "milestone_id"})
        selfs["percentage"] = 1

        # not self loops
        progressions = progressions.query("`from` != `to`")

        # percentage for "from milestone", for start milestone， percentage = 1 - sum(other end milestone percentages). it's important to for divergence region.
        # TODO: for all discrete milestone, progressions group result is empty, but milestone_percentages should not be empty. move it to the nearset milesotne
        froms = progressions.groupby(["cell_id", "from"]).apply(lambda x: 1 - x["percentage"].sum()).rename().reset_index()
        froms.columns = ["cell_id", "milestone_id", "percentage"]

        # percentage for "to milestone", save directly
        tos = progressions[["cell_id", "to", "percentage"]].copy().rename(columns={"to": "milestone_id"})

        milestone_percentages = pd.concat([selfs, froms, tos]).reset_index(drop=True)

        return milestone_percentages

    def group_onto_nearest_milestones(self):
        # TODO: group cells to nearest milestones and get new MilestoneWrapper object
        def get_nearest_milestone(x):
            return x.loc[x["percentage"].idxmax(), "milestone_id"]

        group_df = self.milestone_percentages.groupby("cell_id").apply(get_nearest_milestone)
        milestone_percentages = pd.DataFrame(data={"cell_id": group_df.index, "milestone_id": group_df.values, "percentage": 1.0})
        mw = MilestoneWrapper(
            milestone_network=self.milestone_network,
            milestone_id_list=self.id_list,
            cell_id_list=self.cell_id_list,
            divergence_regions=self.divergence_regions,
            milestone_percentages=milestone_percentages,  # here we use new milestone_percentages and generate
            wrapper_type="cluster",
        )
        return mw

    def group_onto_trajectory_edges(self):
        # TODO: group cells to nearest milestones and get new MilestoneWrapper object
        pass

    def classify_milestone_network(self) -> None:
        """Milestone network classification

        ref: pydynverse/wrap/wrap_add_trajectory.changed_topology
        """
        # TODO: PyDynverse and CFE implementation
        self.milestone_network_class = "N"
        self.directed = False

    # fix for milestone and cell color
    @property
    def milestone_color_dict(self):
        """Lazy load milestone color dictionary."""
        if getattr(self, "_milestone_color_dict", None) is None:
            self._generate_color()
        return self._milestone_color_dict

    @property
    def cell_color_dict(self):
        """Lazy load cell color dictionary."""
        if getattr(self, "_milestone_color_dict", None) is None:
            self._generate_color()
        return self._cell_color_dict

    def _generate_color(self, palette_name=settings.sns_palette, ref_color_dict: dict = None):
        # TODO: auto detect fadata cluster related color for cellrank, scvelo ...
        # color for milestone (rgb).
        if (ref_color_dict is not None) and (set(self.id_list).issubset(set(ref_color_dict.keys()))):
            logger.debug("synchronize milestone color with reference color dict.")
            if isinstance(next(iter(ref_color_dict.values())), str):
                # hex string to rgb list
                def color_func(x):
                    return list(mcolors.to_rgb(x))

            else:
                # rgb list
                def color_func(x):
                    return list(x)

            milestone_color_dict = {milestone_id: color_func(ref_color_dict[milestone_id]) for milestone_id in self.id_list}
        else:
            logger.debug("generate milestone color from palette.")
            n = len(self.id_list)
            palette = sns.color_palette(palette_name)
            if n <= len(palette):
                palette = palette[:n]
            else:
                logger.warning(
                    f"The number of colors({n}) is greater than the number of colors in the '{palette_name}' palette({len(palette)}), and the 'husl' palette selection is used."
                )
                palette = sns.color_palette("husl", n_colors=n)

            milestone_color_list = [list(i) for i in palette]  # transfer from tuple to list, [r, g, b]
            milestone_color_dict = dict(zip(self.id_list, milestone_color_list))
        milestone_color_df = pd.DataFrame(milestone_color_dict, index=["r", "g", "b"]).T

        # color for cell
        def mix_color(mpg):
            # mix related milestone color to get color for a cell
            mpg_color = milestone_color_df.loc[mpg["milestone_id"]]
            mix_color_array = mpg_color.apply(lambda rgb_channel: (rgb_channel.array * mpg["percentage"].array).sum())
            return mcolors.to_hex(mix_color_array)

        cell_color_dict = self.milestone_percentages.groupby("cell_id").apply(lambda mpg: mix_color(mpg)).to_dict()

        self._milestone_color_dict = milestone_color_dict
        self._cell_color_dict = cell_color_dict

    def rename_milestone(self, old2new: dict):
        """
        Rename milestone IDs based on the old2new dictionary, updating all related data structures.

        Parameters:
        - old2new (dict): A dictionary with old milestone IDs as keys and new milestone IDs as values.

        Raises:
        - ValueError: If an old ID does not exist or a new ID already exists.
        """
        # check if old id exists
        all_milestones = set(self.id_list)
        for old_id in old2new.keys():
            if old_id not in all_milestones:
                raise ValueError(f"Old milestone ID '{old_id}' does not exist.")
        # check if new id conflicts
        new_ids = set(old2new.values())
        existing_new_conflicts = new_ids.intersection(all_milestones - set(old2new.keys()))
        if existing_new_conflicts:
            raise ValueError(f"New milestone ID {existing_new_conflicts} already exists.")

        # update milestone id in various attribute
        # list(id_list),
        self.id_list = [old2new.get(mid, mid) for mid in self.id_list]
        # dataframes(milestone_network, milestone_percentages, progressions, divergence_regions)
        self.milestone_network["from"] = self.milestone_network["from"].replace(old2new)
        self.milestone_network["to"] = self.milestone_network["to"].replace(old2new)
        self.milestone_percentages["milestone_id"] = self.milestone_percentages["milestone_id"].replace(old2new)
        self.progressions["from"] = self.progressions["from"].replace(old2new)
        self.progressions["to"] = self.progressions["to"].replace(old2new)
        if hasattr(self, "divergence_regions") and self.divergence_regions is not None and "milestone_id" in self.divergence_regions.columns:
            self.divergence_regions["milestone_id"] = self.divergence_regions["milestone_id"].replace(old2new)
        # dict(_milestone_color_dict and _cell_color_dict)
        if hasattr(self, "_milestone_color_dict") and self._milestone_color_dict is not None:
            self._milestone_color_dict = {old2new.get(k, k): v for k, v in self._milestone_color_dict.items()}
        # if hasattr(self, '_cell_color_dict') and self._cell_color_dict is not None:
        #     pass

        logger.info(f"successfully renamed milestones: {old2new}")

    def subset_by_cells(self, cell_list: list, filter_milestone: bool = False):
        """
        Subset the milestone wrapper by keeping only specified cells.

        Args:
            cell_list (list): A list of cell IDs to keep.

        Returns:
            MilestoneWrapper: A new wrapper object containing the subset.
        """
        # 1. filter milestone_percentages
        sub_percentages = self.milestone_percentages[self.milestone_percentages["cell_id"].isin(cell_list)].copy()
        valid_cells = sub_percentages["cell_id"].unique()

        # 2. filter progressions
        sub_progressions = self.progressions[self.progressions["cell_id"].isin(valid_cells)].copy()

        # 3. filter milestone_network
        if filter_milestone:
            valid_milestones = set(sub_percentages["milestone_id"].unique())
            sub_network = self.milestone_network[
                self.milestone_network["from"].isin(valid_milestones) & self.milestone_network["to"].isin(valid_milestones)
            ].copy()
        else:
            valid_milestones = self.id_list
            sub_network = self.milestone_network

        # 4. filter divergence_regions
        sub_div = pd.DataFrame(columns=self.divergence_regions.columns)
        if hasattr(self, "divergence_regions") and self.divergence_regions is not None and not self.divergence_regions.empty:
            sub_div = self.divergence_regions[self.divergence_regions["milestone_id"].isin(valid_milestones)].copy()

        # 5. filter milestone color dict
        milestone_color_dict = {milestone: self.milestone_color_dict[milestone] for milestone in valid_milestones}

        # 6. create new wrapper
        new_wrapper = MilestoneWrapper(
            milestone_network=sub_network,
            milestone_id_list=list(valid_milestones),
            cell_id_list=list(valid_cells),
            divergence_regions=sub_div,
            milestone_percentages=sub_percentages,
            progressions=sub_progressions,
            wrapper_type=self.wrapper_type if hasattr(self, "wrapper_type") else None,
            name=f"{self.id}_sub",
            milestone_color_dict=milestone_color_dict,
        )
        return new_wrapper

    def subset_by_edges(self, edge_list: list):
        """
        Subset the milestone wrapper by keeping only specified edges.

        Args:
            edge_list (list): A list of tuples, e.g. [('A', 'B'), ('B', 'C')].

        Returns:
            MilestoneWrapper: A new wrapper object containing the subset.
        """
        # 1. filter milestone_network
        # ensure edge_list is a set of tuples for fast lookup
        edge_set = set(tuple(edge) for edge in edge_list)
        # check edges
        self.milestone_network[["from", "to"]]
        # optional_edge_set = set(self.milestone_network.apply(lambda row: (row["from"], row["to"]), axis=1).tolist()))
        optional_edge_set = set([tuple(i) for i in self.milestone_network[["from", "to"]].values.tolist()])
        if len(edge_set & optional_edge_set) == 0:
            # empty intersection
            logger.error("edge set are all invalid, optional valid edge(s): {optional_edge_set}")
        else:
            invalid_edge_set = edge_set - optional_edge_set  # edges are in edges_set but not in optional_edge_set.
            if len(invalid_edge_set) > 0:
                logger.warning(f"edge(s): {invalid_edge_set} is invalid, optional valid edge(s): {optional_edge_set}")
                edge_set = edge_set - invalid_edge_set
        # filter network
        mask_network = self.milestone_network.apply(lambda row: (row["from"], row["to"]) in edge_set, axis=1)
        sub_network = self.milestone_network[mask_network].copy()

        # 2. filter progressions
        mask_prog = self.progressions.apply(lambda row: (row["from"], row["to"]) in edge_set, axis=1)
        sub_progressions = self.progressions[mask_prog].copy()

        valid_cells = sub_progressions["cell_id"].unique()

        # 3. filter samples in milestone_percentages
        sub_percentages = self.milestone_percentages[self.milestone_percentages["cell_id"].isin(valid_cells)].copy()

        # 4. filter divergence_regions
        valid_milestones = set(sub_network["from"]).union(set(sub_network["to"]))
        sub_div = pd.DataFrame(columns=self.divergence_regions.columns)
        if hasattr(self, "divergence_regions") and self.divergence_regions is not None and not self.divergence_regions.empty:
            sub_div = self.divergence_regions[self.divergence_regions["milestone_id"].isin(valid_milestones)].copy()

        # 5. filter milestone color dict
        milestone_color_dict = {milestone: self.milestone_color_dict[milestone] for milestone in valid_milestones}

        # 5. create new wrapper
        new_wrapper = MilestoneWrapper(
            milestone_network=sub_network,
            milestone_id_list=list(valid_milestones),
            cell_id_list=list(valid_cells),
            divergence_regions=sub_div,
            milestone_percentages=sub_percentages,
            progressions=sub_progressions,
            wrapper_type=self.wrapper_type,
            name=f"{self.id}_sub",
            milestone_color_dict=milestone_color_dict,
        )
        return new_wrapper

    def _check_milestone_network(self, milestone_network, default_length=1.0):
        """
        Check the milestone network for invalid values in the "length" column and replace them with the average length.

        Args:
            milestone_network (pd.DataFrame): The milestone network dataframe with a "length" column.

        Returns:
            pd.DataFrame: The validated and corrected milestone network.
        """
        if "length" in milestone_network.columns:
            valid_lengths = milestone_network["length"].replace([np.inf, -np.inf], np.nan).dropna()
            if valid_lengths.empty:
                raise ValueError("All values in the 'length' column are invalid. Cannot compute a valid average.")
            mean_length = valid_lengths.mean()
            if milestone_network["length"].isnull().any():
                logger.warning("milestone_network has missing values in 'length' column, filling with average length.")
                milestone_network["length"].fillna(mean_length, inplace=True)
            if milestone_network["length"].isin([np.inf, -np.inf]).any():
                logger.warning("milestone_network has infinite values in 'length' column, replacing with average length.")
                milestone_network["length"].replace([np.inf, -np.inf], mean_length, inplace=True)
        else:
            milestone_network["length"] = default_length
            logger.debug(f"milestone_network does not have 'length' column, adding with default length({default_length}).")

        return milestone_network

    def _convert_milestone_network_to_graph(self, milestone_network):
        G = nx.from_pandas_edgelist(
            milestone_network,
            source="from",
            target="to",
            edge_attr=True,
            create_using=nx.DiGraph if milestone_network["directed"].any() else nx.Graph,
        )
        #
        discrete_milestones = set(self.id_list) - set(milestone_network["from"]).union(set(milestone_network["to"]))
        for milestone in discrete_milestones:
            G.add_node(milestone)
        return G

    def remove_loop_edges(self):
        # remove loop edge which may generate by fadata.add_trajectory_mannually
        loop_edge_df = self.milestone_network.query("`from` == `to`")
        loop_milestone_list = loop_edge_df["from"].tolist()
        if len(loop_milestone_list) > 0:
            logger.warning(f"remove loop edges for nodes: {loop_milestone_list}")
            self.milestone_network = self.milestone_network.query("`from` != `to`")

            self._milestone_network_G = self._convert_milestone_network_to_graph(self.milestone_network)

    # some basic graph analysis for milestone network, more can be added later
    def is_connected(self):
        if self.directed:
            return nx.is_weakly_connected(self.milestone_network_G)  # DiGraph
        else:
            return nx.is_connected(self.milestone_network_G)  # Graph

    def get_root_milestone(self):
        if self.directed:
            root_milestone_list = [node for node, in_degree in self.milestone_network_G.in_degree() if in_degree == 0]
        else:
            root_milestone_list = [node for node, degree in self.milestone_network_G.degree() if degree == 1]

        # if unconnected graph, return list. if connected graph, return single value.
        return root_milestone_list if len(root_milestone_list) > 1 else root_milestone_list[0]

    # merge operation for milestone or edge
    def merge_edge_trajectory(self, sub_mw: "MilestoneWrapper", replace_edge: str, scale_local_edge_length: bool = True) -> "MilestoneWrapper":
        raise NotImplementedError("merge_edge_trajectory is not implemented yet. Please use merge_milestone_trajectory first.")

    def merge_milestone_trajectory(
        self, sub_mw: "MilestoneWrapper", replace_milestone: str, scale_local_edge_length: bool = True
    ) -> "MilestoneWrapper":
        if sub_mw is None:
            raise ValueError("sub_mw is None")
        if replace_milestone not in set(self.id_list):
            raise ValueError(f"replace_milestone '{replace_milestone}' not found in current milestones")

        sub_mw.remove_loop_edges()

        global_mn = self.milestone_network.copy()
        global_prog = self.progressions.copy()
        local_mn = sub_mw.milestone_network.copy()
        local_prog = sub_mw.progressions.copy()

        # rename conflicts for local milestones (except replacing milestone itself)
        global_nodes = set(global_mn["from"]).union(set(global_mn["to"]))
        local_nodes = set(local_mn["from"]).union(set(local_mn["to"]))
        conflict_nodes = local_nodes.intersection(global_nodes - {replace_milestone})
        rename_map = {m: f"{replace_milestone}::{m}" for m in conflict_nodes}
        if len(rename_map) > 0:
            local_mn["from"] = local_mn["from"].replace(rename_map)
            local_mn["to"] = local_mn["to"].replace(rename_map)
            local_prog["from"] = local_prog["from"].replace(rename_map)
            local_prog["to"] = local_prog["to"].replace(rename_map)

        # local roots/leaves (support connected or disconnected sub graph)
        local_graph = sub_mw.milestone_network_G
        local_directed = sub_mw.directed

        def _to_list(x):
            return x if isinstance(x, list) else [x]

        local_root_list = _to_list(sub_mw.get_root_milestone())
        if local_directed:
            local_leaf_list = [n for n, d in local_graph.out_degree() if d == 0]
        else:
            local_leaf_list = [n for n, d in local_graph.degree() if d <= 1]

        if len(local_root_list) == 0:
            local_root_list = list(local_graph.nodes)
        if len(local_leaf_list) == 0:
            local_leaf_list = list(local_graph.nodes)

        # predecessor and successor edges around replaced milestone in global
        pred_edges = global_mn[global_mn["to"] == replace_milestone].copy()
        succ_edges = global_mn[global_mn["from"] == replace_milestone].copy()

        # optional length scaling to global neighborhood
        if scale_local_edge_length:
            incident_lengths = pd.concat(
                [
                    pred_edges.get("length", pd.Series(dtype=float)),
                    succ_edges.get("length", pd.Series(dtype=float)),
                ]
            )
            incident_lengths = incident_lengths.replace([np.inf, -np.inf], np.nan).dropna()
            target_len = incident_lengths.mean() if len(incident_lengths) > 0 else 1.0

            local_lengths = local_mn.get("length", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
            local_len = local_lengths.mean() if len(local_lengths) > 0 else 1.0
            if local_len == 0:
                local_len = 1.0
            local_mn["length"] = local_mn.get("length", 1.0) * (target_len / local_len)

        # remove global edges touching replace milestone
        kept_global_mn = global_mn[(global_mn["from"] != replace_milestone) & (global_mn["to"] != replace_milestone)].copy()

        # bridge strategy:
        # 1) predecessors -> all local roots (divergence mode for multi-roots)
        # 2) successors are matched to one corresponding local leaf (instead of all-to-all)
        #    matching score = overlap(cell_id) weighted by progression percentages
        bridge_rows = []
        if pred_edges.shape[0] > 1:
            raise Exception(f"The replace milestone has more than 1 predecessor edge {pred_edges}, which is not supported yet.")
        elif pred_edges.shape[0] == 0:
            logger.warning(f"The replace milestone '{replace_milestone}' has no predecessor edge")
        else:
            for local_root in local_root_list:
                bridge_rows.append(
                    {
                        "from": pred_edges["from"].iloc[0],
                        "to": local_root,
                        "length": pred_edges["length"].iloc[0],
                        "directed": self.directed or local_directed,
                    }
                )
        # successor -> matched leaf map
        succ_leaf_map = {}
        if len(succ_edges) > 0:
            succ_target_list = succ_edges["to"].drop_duplicates().tolist()
            for succ in succ_target_list:
                succ_prog = global_prog[(global_prog["from"] == replace_milestone) & (global_prog["to"] == succ)][["cell_id", "percentage"]].copy()
                succ_prog.columns = ["cell_id", "succ_w"]

                best_leaf = None
                best_score = -1.0
                for leaf in local_leaf_list:
                    leaf_prog = local_prog[local_prog["to"] == leaf][["cell_id", "percentage"]].copy()
                    if len(leaf_prog) == 0:
                        continue
                    leaf_prog.columns = ["cell_id", "leaf_w"]
                    merged = succ_prog.merge(leaf_prog, on="cell_id", how="inner")
                    score = float((merged["succ_w"] * merged["leaf_w"]).sum()) if len(merged) > 0 else 0.0
                    if score > best_score:
                        best_score = score
                        best_leaf = leaf

                if best_leaf is None:
                    # fallback: use leaf with max summed local progression percentage
                    leaf_weight = local_prog[local_prog["to"].isin(local_leaf_list)].groupby("to")["percentage"].sum().to_dict()
                    best_leaf = max(local_leaf_list, key=lambda x: leaf_weight.get(x, 0.0))

                succ_leaf_map[succ] = best_leaf

            for _, row in succ_edges.iterrows():
                succ = row["to"]
                leaf = succ_leaf_map[succ]
                bridge_rows.append(
                    {
                        "from": leaf,
                        "to": succ,
                        "length": row["length"] if "length" in row else 1.0,
                        "directed": row["directed"] if "directed" in row else True,
                    }
                )
        if len(bridge_rows) > 0:
            bridge_mn = pd.DataFrame(bridge_rows, columns=["from", "to", "length", "directed"])
        else:
            bridge_mn = pd.DataFrame(columns=["from", "to", "length", "directed"])

        new_mn = pd.concat([kept_global_mn, local_mn, bridge_mn], ignore_index=True)
        new_mn = new_mn.drop_duplicates(subset=["from", "to"], keep="last").reset_index(drop=True)

        # merge progression by cell ownership:
        # - local cells: use local progressions only
        # - non-local cells on predecessor/successor edges touching replace_milestone: remap to bridge edges
        local_cell_set = set(local_prog["cell_id"].unique())
        nonlocal_prog = global_prog[(~global_prog["cell_id"].isin(local_cell_set))].copy()

        # unaffected_global_prog = nonlocal_prog[
        #     (nonlocal_prog["from"] != replace_milestone) & (nonlocal_prog["to"] != replace_milestone)
        # ].copy()
        # pred_related_prog = nonlocal_prog[nonlocal_prog["to"] == replace_milestone].copy()
        # succ_related_prog = nonlocal_prog[nonlocal_prog["from"] == replace_milestone].copy()

        unaffected_global_prog = nonlocal_prog.query("(`from` != @replace_milestone) & (`to` != @replace_milestone)").copy()
        pred_related_prog = nonlocal_prog.query("(`to` == @replace_milestone)").copy()
        succ_related_prog = nonlocal_prog.query("(`from` == @replace_milestone)").copy()

        # helper: map leaf -> nearest reachable root
        root_weight = local_prog[local_prog["from"].isin(local_root_list)].groupby("from")["percentage"].sum().to_dict()
        dominant_root = max(local_root_list, key=lambda x: root_weight.get(x, 0.0)) if len(local_root_list) > 0 else None

        def _leaf_to_root(leaf):
            if leaf is None:
                return dominant_root
            best_root = None
            best_len = np.inf
            for r in local_root_list:
                try:
                    if nx.has_path(local_graph, r, leaf):
                        l = nx.shortest_path_length(local_graph, source=r, target=leaf, weight="length")
                        if l < best_len:
                            best_len = l
                            best_root = r
                except Exception:
                    continue
            return best_root if best_root is not None else dominant_root

        # remap successor-related non-local cells: (replace -> succ) -> (matched_leaf -> succ)
        remap_succ_rows = []
        for _, row in succ_related_prog.iterrows():
            succ = row["to"]
            leaf = succ_leaf_map.get(succ, None)
            if leaf is None:
                leaf = dominant_root if dominant_root is not None else (local_leaf_list[0] if len(local_leaf_list) > 0 else None)
            if leaf is None:
                continue
            remap_succ_rows.append(
                {
                    "cell_id": row["cell_id"],
                    "from": leaf,
                    "to": succ,
                    "percentage": row["percentage"],
                }
            )
        remap_succ_prog = (
            pd.DataFrame(remap_succ_rows, columns=["cell_id", "from", "to", "percentage"])
            if len(remap_succ_rows) > 0
            else pd.DataFrame(columns=["cell_id", "from", "to", "percentage"])
        )

        # remap predecessor-related non-local cells: (pred -> replace) -> (pred -> mapped_root)
        # if a cell also appears on successor edge, infer root from that successor's matched leaf.
        succ_by_cell = succ_related_prog.sort_values("percentage", ascending=False).groupby("cell_id").first()
        remap_pred_rows = []
        for _, row in pred_related_prog.iterrows():
            cid = row["cell_id"]
            pred = row["from"]
            if cid in succ_by_cell.index:
                succ = succ_by_cell.loc[cid, "to"]
                leaf = succ_leaf_map.get(succ, None)
                root = _leaf_to_root(leaf)
            else:
                root = dominant_root
            if root is None:
                continue
            remap_pred_rows.append(
                {
                    "cell_id": cid,
                    "from": pred,
                    "to": root,
                    "percentage": row["percentage"],
                }
            )
        remap_pred_prog = (
            pd.DataFrame(remap_pred_rows, columns=["cell_id", "from", "to", "percentage"])
            if len(remap_pred_rows) > 0
            else pd.DataFrame(columns=["cell_id", "from", "to", "percentage"])
        )

        new_prog = pd.concat([unaffected_global_prog, remap_pred_prog, remap_succ_prog, local_prog], ignore_index=True)

        valid_nodes = set(new_mn["from"]) | set(new_mn["to"])
        new_prog = new_prog[new_prog["from"].isin(valid_nodes) & new_prog["to"].isin(valid_nodes)].copy()
        new_prog = new_prog.drop_duplicates(subset=["cell_id", "from", "to"], keep="last").reset_index(drop=True)

        # divergence_regions strategy:
        # 1) remove ALL global divergence regions related to replace_milestone (same divergence_id group)
        # 2) keep other global divergence regions
        # 3) add local internal divergence regions (with renamed milestone ids)
        # TODO: cells in the removed divergence region should be re-assigned to the new progressions, but this is not supported yet.
        div_cols = ["divergence_id", "milestone_id", "is_start"]

        if hasattr(self, "divergence_regions") and self.divergence_regions is not None and (not self.divergence_regions.empty):
            global_div = self.divergence_regions.copy()
            related_div_ids = set(global_div.loc[global_div["milestone_id"] == replace_milestone, "divergence_id"].tolist())
            if len(related_div_ids) > 0:
                global_div = global_div[~global_div["divergence_id"].isin(related_div_ids)].copy()
            global_div = global_div[global_div["milestone_id"].isin(valid_nodes)].copy()
        else:
            global_div = pd.DataFrame(columns=div_cols)

        if hasattr(sub_mw, "divergence_regions") and sub_mw.divergence_regions is not None and (not sub_mw.divergence_regions.empty):
            local_div = sub_mw.divergence_regions.copy()
            if len(rename_map) > 0:
                local_div["milestone_id"] = local_div["milestone_id"].replace(rename_map)
            local_div = local_div[local_div["milestone_id"].isin(valid_nodes)].copy()
        else:
            local_div = pd.DataFrame(columns=div_cols)

        new_div = pd.concat([global_div, local_div], ignore_index=True)
        if len(new_div) > 0:
            new_div = new_div.drop_duplicates(subset=div_cols).reset_index(drop=True)

        # construct new wrapper, preserve colors where possible
        new_id_list = list(valid_nodes)
        new_color_dict = {}
        if getattr(self, "_milestone_color_dict", None) is not None:
            for k in new_id_list:
                if k in self.milestone_color_dict:
                    new_color_dict[k] = self.milestone_color_dict[k]
        if getattr(sub_mw, "_milestone_color_dict", None) is not None:
            inv_rename = {v: k for k, v in rename_map.items()}
            for k in new_id_list:
                src_k = inv_rename.get(k, k)
                if (k not in new_color_dict) and (src_k in sub_mw.milestone_color_dict):
                    new_color_dict[k] = sub_mw.milestone_color_dict[src_k]

        merged_mw = MilestoneWrapper(
            milestone_network=new_mn,
            milestone_id_list=new_id_list,
            divergence_regions=new_div,
            progressions=new_prog,
            wrapper_type=self.wrapper_type,
            milestone_color_dict=new_color_dict if len(new_color_dict) > 0 else None,
            name=f"{self.id}_merge_{replace_milestone}",
        )
        return merged_mw

    def adjust_edge_length(
        self,
        min_length: float = 0.6,
        max_length: float = 3.0,
        clip_quantile: tuple[float, float] = (0.05, 0.95),
        power: float = 0.75,
        inplace: bool = True,
    ):
        """Adjust edge length according to edge-level cell load.

        Rules:
        1) More cells on an edge -> longer edge.
        2) Cells collapsed on milestone (``from == to``) are evenly distributed to all
           incident milestone edges of that milestone.

        Args:
            min_length (float, optional): Minimum scaled edge length. Defaults to 0.6.
            max_length (float, optional): Maximum scaled edge length. Defaults to 3.0.
            clip_quantile (tuple[float, float], optional): Robust clipping quantiles for
                edge load before normalization. Defaults to (0.05, 0.95).
            power (float, optional): Nonlinear compression factor for normalization.
                Use ``power < 1`` to reduce extreme stretching. Defaults to 0.75.
            inplace (bool, optional): Whether to update ``self.milestone_network``.
                Defaults to True.

        Returns:
            pd.DataFrame | MilestoneWrapper:
                - if ``inplace=True`` return ``self``;
                - else return a new milestone network dataframe with updated ``length``.
        """
        if min_length <= 0 or max_length <= 0:
            raise ValueError("min_length and max_length should be positive.")
        if max_length < min_length:
            raise ValueError("max_length should be greater than or equal to min_length.")
        if not (0 <= clip_quantile[0] <= clip_quantile[1] <= 1):
            raise ValueError("clip_quantile should satisfy 0 <= q_low <= q_high <= 1.")

        mn = self.milestone_network.copy()
        if "length" not in mn.columns:
            mn["length"] = 1.0

        def _edge_key(fr, to):
            return f"{fr}__EDGE__{to}"

        # Build index for existing edges only.
        edge_df = mn[["from", "to"]].copy()
        edge_df["edge_key"] = edge_df.apply(lambda r: _edge_key(r["from"], r["to"]), axis=1)
        edge_load = pd.Series(0.0, index=edge_df["edge_key"].tolist())

        prog = self.progressions.copy()
        if prog.empty:
            logger.warning("progressions is empty; skip edge-length adjustment.")
            return self if inplace else mn

        # 1) Non-self progression contributes to its own edge load.
        non_self = prog.query("`from` != `to`").copy()
        if not non_self.empty:
            non_self["edge_key"] = non_self.apply(lambda r: _edge_key(r["from"], r["to"]), axis=1)
            # weighted by progression percentage as effective cell amount
            non_self_load = non_self.groupby("edge_key")["percentage"].sum()
            common_idx = edge_load.index.intersection(non_self_load.index)
            edge_load.loc[common_idx] = edge_load.loc[common_idx] + non_self_load.loc[common_idx]

        # 2) Self-loop progression (cells collapsed on milestone) is evenly split to
        # all incident edges around that milestone.
        self_loop = prog.query("`from` == `to`").copy()
        if not self_loop.empty:
            milestone_self_load = self_loop.groupby("from")["percentage"].sum().to_dict()
            for milestone, load in milestone_self_load.items():
                incident_mask = (mn["from"] == milestone) | (mn["to"] == milestone)
                incident_edges = mn.loc[incident_mask, ["from", "to"]]
                if incident_edges.empty:
                    continue
                share = float(load) / float(len(incident_edges))
                for fr, to in incident_edges.itertuples(index=False):
                    key = _edge_key(fr, to)
                    if key in edge_load.index:
                        edge_load.loc[key] = float(edge_load.loc[key]) + share

        # Convert load to scaled length.
        load_values = edge_load.values.astype(float)
        if np.allclose(load_values, 0):
            logger.warning("all edge loads are zero; use uniform edge length.")
            scaled_length = pd.Series(min_length, index=edge_load.index)
        else:
            q_low, q_high = np.quantile(load_values, clip_quantile)
            if np.isclose(q_high, q_low):
                norm = np.ones_like(load_values)
            else:
                clipped = np.clip(load_values, q_low, q_high)
                norm = (clipped - q_low) / (q_high - q_low)
            norm = np.power(norm, power)
            scaled = min_length + norm * (max_length - min_length)
            scaled_length = pd.Series(scaled, index=edge_load.index)

        mn["length"] = [float(scaled_length.loc[_edge_key(fr, to)]) for fr, to in zip(mn["from"], mn["to"], strict=False)]

        if inplace:
            self.milestone_network = self._check_milestone_network(mn)
            # refresh cached graph
            self._milestone_network_G = self._convert_milestone_network_to_graph(self.milestone_network)
            return self
        return mn

    def get_milestone_order(
        self,
        root=None,
        order_type="bfs",
    ):
        """Return milestone traversal order for bubble-chart y-axis layout.

        Parameters
        ----------
        root : str or None
            Root milestone.  When None, uses ``get_root_milestone()``.
        order_type : str
            ``"bfs"`` — breadth-first (level-order).
            ``"dfs"`` — depth-first.
            ``"balance"`` — balanced inorder: for each node, larger
            subtrees are placed on the left so that the node sits near
            the centre of its descendant range.  Works on both trees
            and DAGs by constructing a spanning tree first.

        Returns
        -------
        list of str
            Ordered milestone IDs.
        """
        G = self.milestone_network_G
        if root is None:
            root = self.get_root_milestone()

        if order_type == "bfs":
            return list(nx.bfs_tree(G, source=root).nodes())
        elif order_type == "dfs":
            return list(nx.dfs_tree(G, source=root).nodes())
        elif order_type == "balance":
            return self._balanced_inorder_order(G, root)
        else:
            raise ValueError(f"Unknown order_type '{order_type}'. " f"Choose 'bfs', 'dfs', or 'balance'.")

    # ------------------------------------------------------------------
    # Balanced inorder helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _balanced_inorder_order(G, root):
        """Balanced-inorder traversal of a (possibly DAG) milestone graph.

        1. Build a spanning tree rooted at *root* via BFS.
        2. Compute subtree sizes.
        3. For every node with children, sort children by subtree size
           descending, then greedily partition into left / right groups
           so that the current node sits as close to the centre of its
           descendant range as possible.
        4. Traverse: left-group subtrees -> node -> right-group subtrees.
        """
        # 1. spanning tree — covers all nodes reachable from root
        if isinstance(G, nx.DiGraph):
            tree = nx.bfs_tree(G, source=root)
        else:
            tree = nx.bfs_tree(G, source=root)

        reachable = set(tree.nodes())

        # 2. subtree sizes (post-order)
        subtree_size = {}

        def _compute_size(node):
            children = list(tree.successors(node))
            size = 1
            for child in children:
                size += _compute_size(child)
            subtree_size[node] = size
            return size

        _compute_size(root)

        # 3 & 4. balanced inorder recursion
        def _traverse(node):
            children = list(tree.successors(node))  # TODO: keep stable during sorting, refer to self.milestone_id_list
            if not children:
                return [node]

            # sort by subtree size descending
            children.sort(key=lambda c: subtree_size[c], reverse=True)

            # greedy partition into left / right
            left_kids, right_kids = [], []
            left_total, right_total = 0, 0
            for child in children:
                sz = subtree_size[child]
                if left_total <= right_total:
                    left_kids.append(child)
                    left_total += sz
                else:
                    right_kids.append(child)
                    right_total += sz

            result = []
            for child in left_kids:
                result.extend(_traverse(child))
            result.append(node)
            for child in right_kids:
                result.extend(_traverse(child))
            return result

        order = _traverse(root)

        # append any nodes not reachable from root at the end
        extra = [n for n in G.nodes() if n not in reachable]
        return order + extra

    # def group_onto_trajectory_edges(self) -> pd.DataFrame:
    #     """group cells to edges
    #     ref: PyDynverse/pydynverse/wrap/wrap_add_grouping.group_onto_trajectory_edges

    #     Returns:
    #         pd.DataFrame: _description_
    #     """
    #     def get_trajectory_edges(x):
    #         x = x.loc[x["percentage"].idxmax()]
    #         return f"{x['from']}->{x['to']}"
    #     group_df = self.progressions.groupby("cell_id").apply(get_trajectory_edges)
    #     return group_df

    # def group_onto_nearest_milestones(self) -> pd.DataFrame:
    #     """ group cells to nearest milestones
    #     ref: PyDynverse/pydynverse/wrap/wrap_add_grouping.group_onto_nearest_milestones

    #     Returns:
    #         pd.DataFrame: _description_
    #     """

    #     def get_nearest_milestone(x):
    #         return x.loc[x["percentage"].idxmax(), "milestone_id"]
    #     group_df = self.milestone_percentages.groupby("cell_id").apply(get_nearest_milestone)
    #     return group_df

    # def gather_cells_at_milestones(self) -> None:
    #     """Move cells to their nearest milestone

    #     ref: pydynverse/wrap/wrap_gather_cells_at_milestones.gather_cells_at_milestones
    #     """
    #     # gather all cells to their nearest milestone
    #     pass


# TODO: read and write h5ad automatically. However, it will be error if the h5ad file is loaded in a new environment without cef moudle loaded.
# attributes need to be read and written
# attribute_name_list = [
#     "milestone_network",
#     "cell_id_list",
#     "divergence_regions",
#     "milestone_percentages",
#     "progressions",
#     "name",
# ]


# @_REGISTRY.register_write(dest_type=h5py.Group, src_type=MilestoneWrapper, spec=IOSpec("MilestoneWrapper", "0.1.0"))
# def write_milestone_wrapper(
#     f: GroupStorageType,
#     k: str,
#     milestone_wrapper: MilestoneWrapper,
#     *,
#     _writer: Writer,
#     dataset_kwargs: Mapping[str, Any] = MappingProxyType({}),
# ):
#     # create h5 key and save for MilestoneWrapper and WaypointWrapper
#     # ref：https://github.com/scverse/anndata/blob/main/src/anndata/_io/specs/methods.py write_anndata
#     # print(f"write_milestone_wrapper")
#     g = f.require_group(k)
#     for attribute_name in attribute_name_list:
#         attribute = getattr(milestone_wrapper, attribute_name, None)
#         if attribute is not None:
#             _writer.write_elem(g, attribute_name, attribute, dataset_kwargs=dataset_kwargs)


# @_REGISTRY.register_read(h5py.Group, IOSpec("MilestoneWrapper", "0.1.0"))
# def read_milestone_wrapper(elem: GroupStorageType, *, _reader: Reader):
#     # read and create MilestoneWrapper object
#     # ref：https://github.com/scverse/anndata/blob/main/src/anndata/_io/specs/methods.py read_anndata
#     print("read_milestone_wrapper")
#     d = {}
#     for attribute_name in attribute_name_list:
#         if attribute_name in elem:
#             d[attribute_name] = _reader.read_elem(elem[attribute_name])
#     # TODO: create object by __new__ function, add attribute mannualy
#     mw = MilestoneWrapper.__new__(MilestoneWrapper)
#     for k, v in d.items():
#         setattr(mw, k, v)
#     return mw

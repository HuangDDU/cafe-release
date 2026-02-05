# from collections.abc import Mapping
# from types import MappingProxyType
# from typing import Any

# import h5py
import matplotlib.colors as mcolors
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
            milestone_network (pd.DataFrame): milestone network with column list: ["from", "to", "length", "directed"]
            id_list(list): milstone id list, should be specified if there is a discrete milestone
            divergence_regions (pd.DataFrame, optional): divergence regions with column list: ["divergence_id", "milestone_id", "is_start"].
            milestone_percentages (pd.DataFrame, optional): milestone percentage with column list: ["cell_id", "milestone_id", "percentage"].
            progressions (pd.DataFrame, optional): progressions with  column list: ["cell_id", "from", "to", "percentage"].
            name (str, optional): name of the wrapper.

        Raises:
            ValueError: Exactly one of milestone_percentages or progressions, must be defined, the other should be None
        """
        self.id = random_time_string(name)
        self.milestone_network = milestone_network
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
        # TODO: for all discrete milestone, progresions group result is empty.
        # print(progressions.groupby(["cell_id", "from"]).apply(lambda x: 1 - x["percentage"].sum()))
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
            wrapper_type=self.wrapper_type,
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
        # optional_edge_set = set(self.milestone_network.apply(lambda row: (row["from"], row["to"]), axis=1).tolist())
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

import pandas as pd

from .._logging import logger
from ..util import random_time_string
from .fate_wrapper import FateWrapper


class MilestoneWrapper(FateWrapper):
    """Wrapper for trajectory milestones
    """

    def __init__(
        self,
        milestone_network: pd.DataFrame,
        cell_id_list: list = None,
        divergence_regions: pd.DataFrame = None,
        milestone_percentages: pd.DataFrame = None,
        progressions: pd.DataFrame = None,
        name="MilestoneWrapper"
    ):
        """Initialize the MilestoneWrapper class.

        Args:
            milestone_network (pd.DataFrame): milestone network with column list: ["from", "to", "length", "directed"]
            divergence_regions (pd.DataFrame, optional): divergence regions with column list: ["divergence_id", "milestone_id", "is_start"].
            milestone_percentages (pd.DataFrame, optional): milestone percentage with column list: ["cell_id", "milestone_id", "percentage"].
            progressions (pd.DataFrame, optional): progressions with  column list: ["cell_id", "from", "to", "percentage"].
            name (str, optional): name of the wrapper.

        Raises:
            ValueError: Exactly one of milestone_percentages or progressions, must be defined, the other should be None
        """
        self.id = random_time_string(name)
        self.milestone_network = milestone_network
        self.id_list = milestone_network[["from", "to"]].stack().unique().tolist()

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
        if progressions is None:
            # milestone_percentages -> progressions, 'add_trajectory' test case
            progressions = MilestoneWrapper.convert_milestone_percentages_to_progressions(milestone_network, milestone_percentages)
        else:
            # progressions -> milestone_percentages, 'add_trajectory_branch' test case
            milestone_percentages = MilestoneWrapper.convert_progressions_to_milestone_percentages(milestone_network, progressions)
        if cell_id_list is not None:
            self.cell_id_list = cell_id_list
        elif milestone_percentages is not None:
            self.cell_id_list = milestone_percentages["cell_id"].unique().tolist()
        else:
            self.cell_id_list = progressions["cell_id"].unique().tolist()
        self.milestone_percentages = milestone_percentages
        self.progressions = progressions

        # self.classify_milestone_network()
        self.milestone_network_class = "N"
        self.directed = milestone_network["directed"].any()

    @staticmethod
    def convert_milestone_percentages_to_progressions(
        milestone_network: pd.DataFrame,
        milestone_percentages: pd.DataFrame
    ) -> pd.DataFrame:
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
        df2 = pd.merge(df1, milestone_percentages[["cell_id", "milestone_id"]], left_on=["from", "cell_id"], right_on=["milestone_id", "cell_id"])
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
    def convert_progressions_to_milestone_percentages(
        milestone_network: pd.DataFrame,
        progressions: pd.DataFrame
    ) -> pd.DataFrame:
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
        froms = progressions.groupby(["cell_id", "from"]).apply(lambda x: 1- x["percentage"].sum()).rename().reset_index()
        froms.columns = ["cell_id", "milestone_id","percentage"]

        # percentage for "to milestone", save directly
        tos = progressions[["cell_id", "to", "percentage"]].copy().rename(columns={"to": "milestone_id"})

        milestone_percentages = pd.concat([selfs, froms, tos]).reset_index(drop=True)

        return milestone_percentages

    def classify_milestone_network(self) -> None:
        """Milestone network classification

        ref: pydynverse/wrap/wrap_add_trajectory.changed_topology
        """
        # TODO: PyDynverse and CFE implementation
        self.milestone_network_class = "N"
        self.directed = False

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

    def gather_cells_at_milestones(self) -> None:
        """Move cells to their nearest milestone

        ref: pydynverse/wrap/wrap_gather_cells_at_milestones.gather_cells_at_milestones
        """
        # gather all cells to their nearest milestone
        pass

    # def to_hdf5(self, group):
    #     # TODO : complete the write hdf5 implementation
    #     ds = group.create_dataset("data", data=None)
    #     # for k, v in self.items():
    #     #     group.create_dataset(k, data=v)

    # @classmethod
    # def from_hdf5(cls, group):
    #     # TODO : complete the read hdf5 implementation
    #     data = group["data"][:]
    #     return cls(data)

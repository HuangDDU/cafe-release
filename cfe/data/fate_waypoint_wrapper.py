from typing import Callable

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from sklearn.metrics.pairwise import pairwise_distances

from ..util import random_time_string
from .fate_wrapper import FateWrapper
from .fate_milestone_wrapper import MilestoneWrapper


class WaypointWrapper(FateWrapper):
    """Wrapper for trajectory waypoint
    """

    def __init__(
        self,
        milestone_wrapper: MilestoneWrapper,
        name: str = "WaypointWrapper",
        n_waypoints: int = 200,
        transform: Callable[[float], float] = lambda x: x,  # edge length transform function
        resolution: float = None
    ):
        """Initialize the WaypointWrapper class.

        Args:
            milestone_wrapper (MilestoneWrapper): MlestoneWrapper object for trajectory
            name (str, optional): name of the wrapper.
            n_waypoints (int, optional): num of waypoint.
            transform (_type_, optional): transform function for milestone network edge length.
            resolution (float, optional): resolution.
        """
        self.id = random_time_string(name)
        self.milestone_wrapper = milestone_wrapper
        self._select_waypoints(n_waypoints, transform, resolution)

    def _select_waypoints(
            self,
            n_waypoints: int = 200,
            transform: Callable[[float], float] = lambda x: x,  # edge length transform function
            resolution: float = None) -> None:
        """select waypoints base milestone network edge length and resolution parameter

        ref: pydynverse/wrap/wrap_add_waypoints.select_waypoints

        Args:
            n_waypoints (int, optional): num of waypoint.
            transform (_type_, optional): transform function for milestone network edge length.
            resolution (float, optional): resolution.
        """
        mr = self.milestone_wrapper

        if resolution is None:
            # compute resolution automaticall based on the sum of milestone network length after transformation
            resolution = mr.milestone_network["length"].apply(lambda x: transform(x)).sum() / n_waypoints

        # percentage list construction and explode
        def waypoint_id_from_progressions_row(row):
            # get waypoint_id by considering a row comprehensively
            match row["percentage"]:
                case 0:
                    return f"MILESTONE_BEGIN_W{row['from']}_{row['to']}"
                case 1:
                    return f"MILESTONE_END_W{row['from']}_{row['to']}"
                case _:
                    return f"W{row.name+1}"  # waypoint id start from 1
        waypoint_progressions = mr.milestone_network.copy()
        waypoint_progressions["percentage"] = waypoint_progressions["length"].apply(lambda x: [i / x for i in np.arange(0, x, resolution)] + [1])
        waypoint_progressions = waypoint_progressions[["from", "to", "percentage"]]
        waypoint_progressions = waypoint_progressions.explode("percentage").reset_index(drop=True)
        waypoint_progressions["percentage"] = waypoint_progressions["percentage"].astype("float")
        waypoint_progressions["waypoint_id"] = waypoint_progressions.apply(waypoint_id_from_progressions_row, axis=1)
        self.waypoint_progressions = waypoint_progressions

        self.id_list = waypoint_progressions["waypoint_id"].unique().tolist()

        # progressions -> percentages
        waypoint_progressions_tmp = waypoint_progressions.copy()
        waypoint_progressions_tmp = waypoint_progressions_tmp.rename(columns={"waypoint_id": "cell_id"})  # reuse pre column name
        # tmp "cell_id" column name for MilestoneWrapper.reuse convert_progressions_to_milestone_percentages
        waypoint_milestone_percentages = MilestoneWrapper.convert_progressions_to_milestone_percentages(
            milestone_network=mr.milestone_network,
            progressions=waypoint_progressions_tmp
        ).rename(columns={"cell_id": "waypoint_id"})
        self.waypoint_milestone_percentages = waypoint_milestone_percentages

        self.waypoint_geodesic_distances = self._calculate_geodesic_distances().loc[waypoint_progressions["waypoint_id"]]

        waypoint_network = waypoint_progressions\
            .sort_values(by=["from", "to", "percentage"])\
            .groupby(["from", "to"])\
            .apply(lambda group: group.assign(
                from_waypoint=group["waypoint_id"],
                to_waypoint=group["waypoint_id"].shift(-1),
            ))\
            .dropna()\
            .reset_index(drop=True)  # Sort in ascending percentage within the group. "lead" function get the next row, get None if is the last row in group
        waypoint_network = waypoint_network[["from_waypoint", "to_waypoint", "from", "to"]]
        waypoint_network.columns = ["from", "to", "from_milestone_id", "to_milestone_id"]
        self.waypoint_network = waypoint_network

        waypoints = waypoint_milestone_percentages.iloc[waypoint_milestone_percentages.groupby("waypoint_id")["percentage"].idxmax()].reset_index(drop=True)
        waypoints["milestone_id"] = waypoints.apply(lambda x: x["milestone_id"] if x["percentage"] == 1 else None, axis=1)  # if waypoint is not on milestone, the milestone_id=None
        waypoints = waypoints[["waypoint_id", "milestone_id"]]
        self.waypoints = waypoints

    def _calculate_geodesic_distances(self) -> pd.DataFrame:
        """Calculate geodesic distances between cells and waypoints/milestones

        overall idea:
            1. calculate the full path of the target point within each divergent region separately
            2. merge and calculate the distance on the overall graph

        ref: pydynverse/wrap/calculate_geodesic_distances.py

        Returns:
            pd.DataFrame: distances dataframe
        """
        # attribute in the MilestoneWrapper
        cell_id_list = self.milestone_wrapper.cell_id_list
        milestone_id_list = self.milestone_wrapper.id_list
        milestone_network = self.milestone_wrapper.milestone_network
        milestone_percentages = self.milestone_wrapper.milestone_percentages
        divergence_regions = self.milestone_wrapper.divergence_regions
        directed = self.milestone_wrapper.directed

        waypoint_id_list = self.id_list
        waypoint_milestone_percentages = self.waypoint_milestone_percentages

        milestone_percentages = pd.concat([
            milestone_percentages,
            waypoint_milestone_percentages.rename(columns={"waypoint_id": "cell_id"})
        ])

        # remae all milestone ids to MILESTONE_ID
        def milestone_trafo_fun(x):
            #可能的补丁如下，这里我加了一个检查
            x = str(x)
            if x.startswith("MILESTONE_"):
                return x
            return f"MILESTONE_{x}"
        milestone_network = milestone_network.copy()  # don't affect the original milestone_network
        milestone_network["from"] = milestone_network["from"].apply(milestone_trafo_fun)
        milestone_network["to"] = milestone_network["to"].apply(milestone_trafo_fun)
        milestone_id_list = list(map(milestone_trafo_fun, milestone_id_list))
        milestone_percentages["milestone_id"] = milestone_percentages["milestone_id"].apply(milestone_trafo_fun)
        divergence_regions["milestone_id"] = divergence_regions["milestone_id"].apply(milestone_trafo_fun)

        # add an extra divergence area, where normal edges are also treated as divergence areas
        extra_divergences = milestone_network.copy()
        extra_divergences = extra_divergences[~(extra_divergences["from"] == extra_divergences["to"])]
        # extra_divergences = extra_divergences.query("not from == to") # query is more elegant
        # in_divergence determines whether the current edge is within the existing divergence region
        divergence_regions_set_list = divergence_regions.groupby("divergence_id")["milestone_id"].apply(set).tolist()

        def is_milestone_in_divergence(milestone_set, divergence_regions_set_list):
            for divergence_regions_set in divergence_regions_set_list:
                if milestone_set.issubset(divergence_regions_set):
                    return True
            return False
        extra_divergences["in_divergence"] = extra_divergences.apply(lambda x: is_milestone_in_divergence({x["from"], x["to"]}, divergence_regions_set_list), axis=1)
        extra_divergences = extra_divergences[~extra_divergences["in_divergence"]]  # only reserve the new divergence area
        extra_divergences["divergence_id"] = extra_divergences.apply(lambda x: f"{x['from']}__{x['to']}", axis=1)
        extra_divergences = pd.concat([
            # add new columns: milestone_id, is_start
            extra_divergences.assign(milestone_id=extra_divergences["from"], is_start=True),
            extra_divergences.assign(milestone_id=extra_divergences["to"], is_start=False)
        ])[["divergence_id", "milestone_id", "is_start"]]

        # merge divergence regions
        divergence_regions = pd.concat([divergence_regions, extra_divergences]).reset_index(drop=True)
        divergence_ids = divergence_regions["divergence_id"].unique()

        # NetworkX for related data from edge DataFrame
        milestone_graph = nx.from_pandas_edgelist(milestone_network, source="from", target="to", edge_attr="length")
        divergence_regions["is_start"] = divergence_regions["is_start"].astype(bool)  # ensure "is_start" column is bool

        # NOTE: 1. Calculate separately
        # calculate the distance between cells within the divergent
        def calc_divergence_inner_distance_df(did):
            dir = divergence_regions[divergence_regions["divergence_id"] == did]
            mid = dir[dir["is_start"]]["milestone_id"].tolist()  # starting point of the region is milestone_id
            tent = dir["milestone_id"].tolist()  # milestone_id of all milestones in the divergence
            tent_distances = pd.DataFrame(index=mid, columns=tent, data=np.zeros((len(mid), len(tent))))  # The distance from the starting point within the region to all milestones
            # extract corresponding edges from the graph
            for i in mid:
                for j in tent:
                    if i == j:
                        tent_distances.loc[i, j] = 0
                    else:
                        tent_distances.loc[i, j] = milestone_graph.edges[(i, j)]["length"]
            # find cell_id of relevant points by reusing is_milestone_in_divergence
            relevant_pct_cell_id_list = milestone_percentages.groupby("cell_id")["milestone_id"].apply(lambda x: is_milestone_in_divergence(set(x), [set(tent)]))
            relevant_pct_cell_id_list = relevant_pct_cell_id_list[relevant_pct_cell_id_list].index.to_list()
            relevant_pct = milestone_percentages[milestone_percentages["cell_id"].apply(lambda x: x in relevant_pct_cell_id_list)]
            if relevant_pct.shape[0] <= 1:
                return None

            scaled_dists = relevant_pct.copy()
            scaled_dists["dist"] = scaled_dists.apply(lambda x: x["percentage"] * tent_distances.loc[mid, x["milestone_id"]], axis=1)
            tent_distances_long = tent_distances.melt(var_name="from", value_name="length")  # wide data to long data
            tent_distances_long["to"] = tent_distances_long["from"]

            pct_mat = pd.concat([
                scaled_dists[["cell_id", "milestone_id", "dist"]].rename(columns={"cell_id": "from", "milestone_id": "to", "dist": "length"}),
                tent_distances_long
            ])\
                .drop_duplicates()\
                .pivot(index="from", columns="to", values="length").fillna(0)  # (n_cell+n_milestone+n_waypoint)*n_milestone, long data to wide data, "from" is index

            wp_cells = list(set(pct_mat.index) & set(waypoint_id_list))

            if directed:
                # TODO: directed graph
                pass

            distances = pairwise_distances(pct_mat, pct_mat.loc[wp_cells + tent], metric="manhattan")
            distances = pd.DataFrame(index=pct_mat.index, columns=wp_cells + tent, data=distances)
            distances = distances.reset_index().melt(id_vars="from", var_name="to", value_name="length")  # wide data to long data
            distances = distances[~(distances["from"] == distances["to"])]
            return distances

        cell_in_tent_distances = pd.concat([calc_divergence_inner_distance_df(did) for did in divergence_ids])

        if directed:
            # TODO: directed graph
            pass

        # NOTE: 2. merge calculation(use igraph to accelerate compared to networkx)
        # select the shortest distance mode for subsequent directed graphs
        if directed or directed == "forward":
            mode = "out"
        elif directed == "reverse":
            mode = "in"
        else:
            mode = "all"
        # extract the shortest edge after merging, currently has little effect, may be useful for the ring graph
        edgelist_df = pd.concat([milestone_network, cell_in_tent_distances]).groupby(["from", "to"]).agg({"length": "min"}).reset_index()
        # merge two graph to one graph
        gr = ig.Graph.TupleList(edgelist_df.values, edge_attrs=["length"])
        shortest_paths = gr.shortest_paths(source=waypoint_id_list, target=cell_id_list, weights="length", mode=mode)
        out = pd.DataFrame(shortest_paths, index=waypoint_id_list, columns=cell_id_list)

        # TODO: filter cells
        cell_ids_filtered_list = []
        if len(cell_ids_filtered_list) > 0:
            pass

        return out.loc[waypoint_id_list, cell_id_list]

import numpy as np
import pandas as pd
import networkx as nx
import anndata as ad
import scanpy as sc

from .._logging import logger
from ..util import random_time_string

from .fate_milestone_wrapper import MilestoneWrapper
from .fate_waypoint_wrapper import WaypointWrapper


class FateAnnData(ad.AnnData):
    """AnnData object for CellFateExplorer, related data are stored in the object.uns["cfe"] attribute.
    """

    def __init__(
            self,
            name: str = "FateAnnData",
            *args,
            **kwargs
    ):
        """Initialize the FateAnnData class.

        Args:
            name (str, optional): name of the FateAnnData object.
        """
        # logger.debug("FateAnnData __init__")
        self.id = random_time_string(name)
        super().__init__(*args, **kwargs)

        cfe_dict = self.uns.get("cfe", {})  # try to get the stored FateAnnData information

        self.prior_information = cfe_dict.get("prior_information", {})
        cfe_dict["prior_information"] = self.prior_information

        # milestone_wrapper and waypoint_wrapper for latest model
        self.model_name = cfe_dict.get("model_name", "default")

        # milestone_wrapper and waypoint_wrapper for all model
        if "trajectory_history_dict" not in cfe_dict:
            cfe_dict["trajectory_history_dict"] = {}
        self.trajectory_history_dict = cfe_dict.get("trajectory_history_dict", {})

        # NOTE: Other attributes will be added later.
        self.is_wrapped_with_trajectory = False
        self.is_wrapped_with_waypoints = False

        self.cfe_dict = cfe_dict
        self.uns["cfe"] = self.cfe_dict

    @property
    def milestone_wrapper(self):
        # return self._milestone_wrapper
        model_dict = self.cfe_dict["trajectory_history_dict"].get(self.model_name, None)
        if model_dict is not None:
            return model_dict.get("milestone_wrapper")
        else:
            return None

    @milestone_wrapper.setter
    def milestone_wrapper(self, value):
        # self._milestone_wrapper = value
        model_dict = self.cfe_dict["trajectory_history_dict"].get(self.model_name, None)
        if model_dict is not None:
            model_dict["milestone_wrapper"] = value
        else:
            self.cfe_dict["trajectory_history_dict"][self.model_name] = {"milestone_wrapper": value}

    @property
    def waypoint_wrapper(self):
        # return self._waypoint_wrapper
        model_dict = self.cfe_dict["trajectory_history_dict"].get(self.model_name, None)
        if model_dict is not None:
            return model_dict.get("waypoint_wrapper")
        else:
            return None

    @waypoint_wrapper.setter
    def waypoint_wrapper(self, value):
        # self._waypoint_wrapper = value
        model_dict = self.cfe_dict["trajectory_history_dict"].get(self.model_name, None)
        if model_dict is not None:
            model_dict["waypoint_wrapper"] = value
        else:
            self.cfe_dict["trajectory_history_dict"][self.model_name] = {"waypoint_wrapper": value}

    @classmethod
    def from_anndata(cls, adata: ad.AnnData) -> "FateAnnData":
        """Create a FateAnnData object from an existing AnnData object.

        Args:
            adata (ad.AnnData): existing AnnData object

        Returns:
            fadata (cfe.data.FateAnnData): generated FateAnnData object
        """

        logger.debug("Create a FateAnnData object from an existing AnnData object.")

        fadata = cls(
            name=adata.name if hasattr(adata, "name") else "FateAnnData",
            X=adata.X,
            obs=adata.obs,
            var=adata.var,
            uns=adata.uns,
            obsm=adata.obsm,
            varm=adata.varm,
            layers=adata.layers
        )

        return fadata

    @classmethod
    def read_dynverse_simulation_data(
        cls,
        data_filename="synthetic/dyntoy/bifurcating_1.rds",
        data_dir="/usr/share/CellFateExplorer/dynbenchmark/data/"
    ):
        # read dynverse simulation data and create FateAnnData object, default data dir is in /usr/share/CellFateExplorer/dynbenchmark/data/
        import rpy2.robjects as ro
        from ..util import rpy2_read  # rpy2 data structure transfer automatically

        r_script = f"""
        dataset <- readRDS("{data_dir}/{data_filename}")
        dataset
        """
        dataset = ro.r(r_script)

        # crreate FateAnnData object base expression and count matrix
        layers = {}
        if "expression" in dataset:
            X = dataset["expression"]
            layers["expression"] = dataset["expression"]
        if "counts" in dataset:
            X = dataset["counts"]
            layers["counts"] = dataset["counts"]
        fadata = cls(name=dataset["id"], X=X)
        fadata.layers = layers

        # other Anndata attributes
        # if dataset.has_key("cell_info"):
        #     fadata.obs = dataset["cell_info"]
        fadata.obs = dataset.get("cell_info", fadata.obs)  # equal to above
        fadata.obs.index = dataset["cell_ids"]
        fadata.var = dataset.get("feature_info", fadata.obs)
        fadata.var.index = dataset.get("feature_ids", fadata.var.index)

        # call FateAnnData object method
        if "prior_information" in dataset:
            fadata.add_prior_information(**dataset["prior_information"])
        if "milestone_network" in dataset:
            milestone_network = dataset["milestone_network"]
            milestone_percentages = dataset["milestone_percentages"]
            divergence_regions = dataset["divergence_regions"]
            progressions = dataset["progressions"]
            fadata.add_model_name("ref")
            fadata.add_trajectory(
                milestone_network=milestone_network,
                divergence_regions=divergence_regions,
                milestone_percentages=milestone_percentages,
                # progressions=progressions # may cover milestone_percentages
            )

        if "grouping" in dataset:
            fadata.obs["grouping"] = pd.Categorical(dataset["grouping"], dataset["group_ids"])
        # TODO: waypoint add
        return fadata

    def add_prior_information(self, **kwargs) -> None:
        """Add prior information to the FateAnnData object.

        ref: pydynverse/wrap/wrap_add_prior_information add_prior_information
        """
        self.prior_information.update(kwargs)

    def add_model_name(self, model_name: str):
        self.model_name = model_name
        self.cfe_dict["model_name"] = model_name
        self.trajectory_history_dict[self.model_name] = {}

    def get_all_model_name(self, parse=True):
        from ..util import parse_random_time_string

        model_name_list = list(self.trajectory_history_dict.keys())
        if self.model_name not in self.trajectory_history_dict:
            model_name_list = [self.model_name] + model_name_list
        if parse:
            # parse model_name from random_time_string
            model_name_list = [parse_random_time_string(i) for i in model_name_list]
        return model_name_list

    def add_trajectory(
        self,
        milestone_network: pd.DataFrame,
        divergence_regions: pd.DataFrame = None,
        milestone_percentages: pd.DataFrame = None,
        progressions: pd.DataFrame = None,
    ) -> None:
        """Create MilestoneWrapper object as trajectory

        Args:
            milestone_network (pd.DataFrame): milestone network with column list: ["from", "to", "length", "directed"]
            divergence_regions (pd.DataFrame, optional): divergence regions with column list: ["divergence_id", "milestone_id", "is_start"].
            milestone_percentages (pd.DataFrame, optional): milestone percentage with column list: ["cell_id", "milestone_id", "percentage"].
            progressions (pd.DataFrame, optional): progressions with column list: ["cell_id", "from", "to", "percentage"].
        """

        logger.debug("FateAnnData add_trajectory")

        milestone_wrapper = MilestoneWrapper(
            milestone_network=milestone_network,
            cell_id_list=self.obs.index,
            divergence_regions=divergence_regions,
            milestone_percentages=milestone_percentages,
            progressions=progressions
        )

        # if self.milestone_wrapper is not None:
        #     trajectory_history = {}
        #     trajectory_history["milestone_wrapper"] = self.milestone_wrapper
        #     if self.waypoint_wrapper is not None:
        #         trajectory_history["waypoint_wrapper"] = self.waypoint_wrapper
        #     self.trajectory_history_dict[self.model_name] = trajectory_history

        self.milestone_wrapper = milestone_wrapper
        # TODO: save multiple trajectory in cfe_dict
        self.is_wrapped_with_trajectory = True

        if self.model_name not in self.trajectory_history_dict:
            self.trajectory_history_dict[self.model_name] = {}
        self.trajectory_history_dict[self.model_name]["milestone_wrapper"] = milestone_wrapper

    def add_trajectory_mannually(
            self,
            milestone_network: pd.DataFrame,
            cluster_key: str = "clusters",
            basis: str = "X_umap",
            distance_metric: str = "euclidean",
            model_name: str = "ref"
    ):
        """add trajectory mannually as ref trajectory, reuse add_trajectory_projection to get progression

        Args:
            milestone_network (pd.DataFrame): milestone network
            cluster_key (str, optional): _description_. Defaults to "clusters".
            basis (str, optional):cell embedding key. Defaults to "X_umap".
            distance_metric (str, optional): distance metric. Defaults to "euclidean".
            model_name (str, optional): _description_. Defaults to "ref".
        """
        # TODO: add divergence

        from sklearn.metrics.pairwise import pairwise_distances

        self.add_model_name(model_name)

        obs = self.obs.reset_index()  # change index
        milestone_id_list = list(obs[cluster_key].cat.categories)
        X_emb = self.obsm[basis]
        milestone_emb = np.array(list(obs.groupby(cluster_key).apply(lambda x: X_emb[list(x.index)].mean(axis=0))))
        milestone_emb = pd.DataFrame(milestone_emb, index=milestone_id_list)
        # self.obs = self.obs.set_index("index")

        # milestone network
        dis = pd.DataFrame(pairwise_distances(milestone_emb, metric=distance_metric), index=milestone_id_list, columns=milestone_id_list)
        milestone_network["length"] = milestone_network.apply(lambda row: dis.loc[row["from"], row["to"]], axis=1)
        milestone_network["directed"] = True

        # progressions
        self.add_trajectory_projection(
            milestone_network=milestone_network,
            milestone_emb=milestone_emb,
            X_emb=X_emb,
            cluster_key=cluster_key
        )

    def add_trajectory_by_type(self, trajectory_dict: dict) -> None:
        wrapper_type = trajectory_dict["wrapper_type"]
        if wrapper_type == "directed":
            self.add_trajectory(**trajectory_dict)
        elif wrapper_type == "branch":
            self.add_trajectory_branch(
                branch_network=trajectory_dict["branch_network"],
                branches=trajectory_dict["branches"],
                branch_progressions=trajectory_dict["branch_progressions"]
            )
        elif wrapper_type == "linear":
            self.add_trajectory_linear(pseudotime=trajectory_dict["pseudotime"])
        elif wrapper_type == "cycle":
            self.add_trajectory_cycle(pseudotime=trajectory_dict["pseudotime"])
        elif wrapper_type == "probability":
            # self.add_trajectory_end_state_probibalities
            self.add_trajectory_probability(
                end_state_probabilities=trajectory_dict["end_state_probabilities"],
                pseudotime=trajectory_dict["pseudotime"],
            )
        elif wrapper_type == "cluster":
            # self.add_trajectory_cluster_graph
            self.add_trajectory_cluster(
                milestone_network=trajectory_dict["milestone_network"],
                cluster=trajectory_dict["cluster"]
            )
        elif wrapper_type == "projection":
            self.add_trajectory_projection(
                milestone_network=trajectory_dict["milestone_network"],
                milestone_emb=trajectory_dict["milestone_emb"],
                X_emb=trajectory_dict["X_emb"],
                cluster_key=trajectory_dict.get("cluster_key", None)
            )
        elif wrapper_type == "graph":
            # self.add_trajectory_cell_graph
            self.add_trajectory_graph(
                cell_graph=trajectory_dict["cell_graph"],
                to_keep=trajectory_dict["to_keep"],
            )

    # def add_trajectory_by_type_bak(self, trajectory_dict: dict) -> None:
    #     """Call the trajectory addition method based on specific trajectory types

    #     Args:
    #         trajectory_dict (dict): trajectory dict result based on specific trajectory types
    #     """

    #     trajectory_dict_keys = trajectory_dict.keys()
    #     # TODO: choose trajectory type automatically by method object from yaml file
    #     if "pseudotime" in trajectory_dict_keys:
    #         # there are 3 wrapper with pseudotime
    #         if "cycle" in trajectory_dict_keys:
    #             # cycle wrapper
    #             self.add_trajectory_cycle(
    #                 pseudotime=trajectory_dict["pseudotime"]
    #             )
    #         elif "end_state_probabilities" in trajectory_dict_keys:
    #             # probibalitiy wrapper
    #             self.add_trajectory_end_state_probibalities(
    #                 end_state_probabilities=trajectory_dict["end_state_probabilities"],
    #                 pseudotime=trajectory_dict["pseudotime"]
    #             )
    #         else:
    #             # linear wrapper
    #             self.add_trajectory_linear(trajectory_dict["pseudotime"])
    #     elif "branch_network" in trajectory_dict_keys:
    #         # branch wrapper
    #         self.add_trajectory_branch(
    #             branch_network=trajectory_dict["branch_network"],
    #             branches=trajectory_dict["branches"],
    #             branch_progressions=trajectory_dict["branch_progressions"]
    #         )

    #     elif "cluster" in trajectory_dict:
    #         # cluster graph
    #         self.add_trajectory_cluster_graph(
    #             milestone_network=trajectory_dict["milestone_network"],
    #             cluster=trajectory_dict["cluster"]
    #         )
    #     elif "milestone_emb" in trajectory_dict_keys:
    #         # projection
    #         self.add_trajectory_projection(
    #             milestone_network=trajectory_dict["milestone_network"],
    #             milestone_emb=trajectory_dict["milestone_emb"],
    #             X_emb=trajectory_dict["X_emb"],
    #             cluster_key=trajectory_dict.get("cluster_key", None)
    #         )
    #     elif "cell_graph" in trajectory_dict_keys:
    #         # cell graph
    #         self.add_trajectory_cell_graph(
    #             cell_graph=trajectory_dict["cell_graph"],
    #             to_keep=trajectory_dict["to_keep"],
    #         )
    #     # elif "velocity" in trajectory_dict_keys:
    #     #     self.add_trajectory_velocity(
    #     #         neighbors=trajectory_dict["neighbors"],
    #     #         velocity=trajectory_dict["velocity"],
    #     #         velocity_graph=trajectory_dict["velocity_graph"],
    #     #     )
    #     else:
    #         # defult direct output
    #         self.add_trajectory(**trajectory_dict)

    def add_waypoints(self, milestone_wrapper: MilestoneWrapper = None) -> None:
        """Create WaypointWrapper object

        """
        logger.debug("FateAnnData add_waypoints")
        milestone_wrapper = milestone_wrapper if milestone_wrapper is not None else self.milestone_wrapper  # waypoint is based on milestone
        waypoint_wrapper = WaypointWrapper(milestone_wrapper)
        # waypoint_wrapper.waypoint_geodesic_distances = waypoint_wrapper.waypoint_geodesic_distances.loc[:,self.obs.index] #
        self.waypoint_wrapper = waypoint_wrapper
        self.cfe_dict["waypoint_wrapper"] = waypoint_wrapper
        self.is_wrapped_with_waypoints = True

        if self.model_name not in self.trajectory_history_dict:
            self.trajectory_history_dict[self.model_name] = {}
        self.trajectory_history_dict[self.model_name]["waypoint_wrapper"] = waypoint_wrapper

    def add_trajectory_branch(
            self,
            branch_network: pd.DataFrame,
            branch_progressions: pd.DataFrame,
            branches: pd.DataFrame
    ) -> None:
        """Add branch trajectory,such as PAGA

        ref: PyDynverse/pydynverse/wrap/wrap_add_branch_trajectory.add_branch_trajectory

        Args:
            branch_network (pd.DataFrame): branch network with column list: ["from", "to"]
            branch_progressions (pd.DataFrame): branch progressions with column list: ["cell_id", "branch_id", "percentage"
            branches (pd.DataFrame): branches with column list: ["branch_id", "length", "directed"]
        """
        logger.debug("FateAnnData add_trajectory_branch")

        branch_id_list = branches["branch_id"]
        milestone_network = pd.DataFrame({
            "from": map(lambda x: f"{x}_from", branch_id_list),
            "to": map(lambda x: f"{x}_to", branch_id_list),
            "branch_id": branch_id_list
        })
        milestone_mapper_network = pd.concat(
            [
                # single from node
                pd.DataFrame({
                    "from": map(lambda x: f"{x}_from", branch_id_list),
                    "to": map(lambda x: f"{x}_from", branch_id_list),
                }),
                # connected node, if "A->B" in branch_network , then "A_to->B_from" in here,
                pd.DataFrame({
                    "from": map(lambda x: f"{x}_to", branch_network["from"]),
                    "to": map(lambda x: f"{x}_from", branch_network["to"]),
                }),
                # single to node
                pd.DataFrame({
                    "from": map(lambda x: f"{x}_to", branch_id_list),
                    "to": map(lambda x: f"{x}_to", branch_id_list),
                }),
            ])
        # transform node name to connected component id
        mapper = {}
        graph = nx.from_pandas_edgelist(milestone_mapper_network, source="from", target="to")
        connected_components = nx.connected_components(graph)
        for component_index, component in enumerate(connected_components):
            for node in component:
                mapper[node] = str(component_index + 1)  # milestone id starts from 1
        milestone_network["from"] = milestone_network["from"].apply(lambda x: mapper[x])
        milestone_network["to"] = milestone_network["to"].apply(lambda x: mapper[x])
        milestone_network = pd.merge(milestone_network, branches, on="branch_id")

        progressions = pd.merge(branch_progressions, milestone_network, on="branch_id")[["cell_id", "from", "to", "percentage"]]

        milestone_network = milestone_network[["from", "to", "length", "directed"]]

        self.add_trajectory(
            milestone_network=milestone_network,
            progressions=progressions
        )

    def add_trajectory_linear(
        self,
        pseudotime: list,
        directed: bool = False,
        do_scale_minmax: bool = True,
    ) -> None:
        """add linear trajectory, such as Comp1(baseline), Palantir, Cytotrace.

        ref: PyDynverse/pydynverse/wrap/wrap_add_linear_trajector.wrap_add_linear_trajector.add_linear_trajectory

        Args:
            pseudotime (list): pseudotime sequence.
        """
        pseudotime = np.array(pseudotime)

        # min-max scale pseudotime to [0, 1]
        if do_scale_minmax:
            pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
        else:
            assert (pseudotime >= 0).all() and (pseudotime <= 1).all()
        milestone_ids = ["milestone_begin", "milestone_end"]
        # milestone_network datframe construction, length=1
        milestone_network = pd.DataFrame({
            "from": milestone_ids[0],
            "to": milestone_ids[1],
            "length": 1,
            "directed": directed,
        }, index=[0])  # all scalar, need "index" to show sample num
        # progressions datafram construction， percentage=pseudotime
        progressions = pd.DataFrame({
            "cell_id": self.obs.index,
            "from": milestone_ids[0],
            "to": milestone_ids[1],
            "percentage": pseudotime,
        })
        self.add_trajectory(
            milestone_network=milestone_network,
            divergence_regions=None,
            progressions=progressions
        )

    def add_trajectory_cycle(self,
                             pseudotime: list,
                             directed: bool = False,
                             do_scale_minmax: bool = True,
                             ) -> None:
        pseudotime = np.array(pseudotime)

        # min-max scale pseudotime to [0, 1]
        if do_scale_minmax:
            pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
        else:
            assert (pseudotime >= 0).all() and (pseudotime <= 1).all()

        # milestone_network: A->B, B->C, C->A
        milestone_ids = ["A", "B", "C"]
        milestone_network = pd.DataFrame({
            "from": milestone_ids,
            "to": milestone_ids[1:] + [milestone_ids[0]],
            "length": 1,
            "directed": directed,
            "edge_id": range(len(milestone_ids))
        })

        # progression: 3 segement
        progressions = pd.DataFrame({
            "cell_id": self.obs.index,
            "time": [3*i for i in pseudotime],
        })
        progressions["edge_id"] = progressions["time"].apply(lambda x: 0 if x <= 1 else 1 if x <= 2 else 2).astype("int")
        progressions = pd.merge(progressions, milestone_network[["from", "to", "edge_id"]], on="edge_id")
        progressions["percentage"] = progressions["time"] - progressions["edge_id"]
        progressions = progressions[["cell_id", "from", "to", "percentage"]].reset_index(drop=True)

        milestone_network = milestone_network[["from", "to", "length", "directed"]]

        self.add_trajectory(
            milestone_network=milestone_network,
            divergence_regions=None,
            progressions=progressions
        )

    def add_trajectory_probability(
        self,
        end_state_probabilities: pd.DataFrame,
        pseudotime: list,
        do_scale_minmax: bool = True
    ):
        if do_scale_minmax:
            pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
        if end_state_probabilities.shape[1] == 1:
            # 只有一个终端状态，就是线性轨迹了
            trajectory = self.add_trajectory_linear(
                pseudotime=pseudotime,
                directed=True,
                do_scale_minmax=do_scale_minmax,
            )
        else:
            # 多个终端状态， 构建里程碑网络
            start_milestone_id = "milestone_begin"  # 起始点是一个完全虚拟点
            end_milestone_ids = end_state_probabilities.columns[1:].tolist()  # 终端点从列名中提取, 默认第一列为cell_id
            milestone_ids = [start_milestone_id] + end_milestone_ids

            # 起始点作为中心的星型里程碑网络
            milestone_network = pd.DataFrame({
                "from": start_milestone_id,
                "to": end_milestone_ids,
                "length": 1,
                "directed": True
            })

            # 添加发散区域，由所有里程碑节点共同构成构成
            divergence_regions = pd.DataFrame({
                "milestone_id": milestone_ids,
                "divergence_id": "D",
                "is_start": pd.Series(milestone_ids) == start_milestone_id
            })

            pseudotime = pd.Series(pseudotime, index=end_state_probabilities["cell_id"])
            progressions = end_state_probabilities.melt(id_vars=["cell_id"], var_name="to", value_name="percentage")
            progressions["from"] = start_milestone_id
            progressions["percentage"] = progressions.groupby("cell_id")["percentage"].transform(lambda x: x / x.sum() * pseudotime[x.name])  # 缩放使其之和为1，暂时不理解这个
            progressions = progressions[["cell_id", "from", "to", "percentage"]]

            self.add_trajectory(
                milestone_network=milestone_network,
                divergence_regions=divergence_regions,
                progressions=progressions
            )

    def add_trajectory_cluster(
            self,
            milestone_network: pd.DataFrame,
            cluster: str | list,
    ):
        cluster_list = cluster
        mn_ft = milestone_network[["from", "to"]]
        both_direction = pd.concat([
            mn_ft.assign(label=mn_ft["from"], percentage=0),
            mn_ft.assign(label=mn_ft["to"], percentage=1)
        ])

        progressions = pd.DataFrame({"cell_id": self.obs.index, "label": cluster_list})\
            .merge(both_direction, on="label")\
            .groupby("cell_id")\
            .apply(lambda x: x.sort_values("percentage", ascending=False).iloc[0])\
            .reset_index(drop=True)\
            .drop("label", axis=1)

        self.add_trajectory(
            milestone_network=milestone_network,
            divergence_regions=None,
            progressions=progressions,
        )

    def add_trajectory_projection(
            self,
            milestone_network: pd.DataFrame,
            milestone_emb: pd.DataFrame | np.ndarray,
            X_emb: pd.DataFrame | np.ndarray | str,
            cluster_key: str = None,
    ):
        """add projection trajectory, such as MST(baseline)

        ref: PyDynverse/pydynverse/wrap/wrap_add_dimred_projection.add_dimred_projection

        Args:
            milestone_network (pd.DataFrame): milestone network
            milestone_emb (pd.DataFrame | np.ndarray): embbeding for milestones.
            X_emb (pd.DataFrame | np.ndarray | str): embedding for cells.
            cluster_key (str, optional): cluster key.
        """
        from ..util import project_to_segments

        if type(X_emb) == str:
            X_emb = self.obsm[X_emb]
        if not type(X_emb) == pd.DataFrame:
            X_emb = pd.DataFrame(X_emb, index=self.obs.index)

        if cluster_key is None:
            # if no cluster key is given, just project all cells to the segments
            proj = project_to_segments(
                x=X_emb,
                segment_start=milestone_emb.loc[milestone_network["from"],],
                segment_end=milestone_emb.loc[milestone_network["to"],],
            )
            progressions = milestone_network.iloc[proj["segment"]-1][["from", "to"]]
            progressions["cell_id"] = self.obs.index
            progressions["percentage"] = proj["progression"]
            progressions = progressions[["cell_id", "from", "to", "percentage"]].reset_index(drop=True)
        else:
            # project cells onto the line segments corresponding to their respective clusters
            cluster_series = self.obs[cluster_key]
            cluster_id_list = cluster_series.unique()
            progressions = []

            for cluster in cluster_id_list:
                cids = cluster_series[cluster_series == cluster].index
                if cids.shape[0] > 0:
                    # project to segments
                    mns = milestone_network.query("`from` == @cluster or `to` == @cluster")  # query，`` cloumn，@ value
                    if mns.shape[0] > 0:
                        proj = project_to_segments(
                            x=X_emb.loc[cids],
                            segment_start=milestone_emb.loc[mns["from"],],
                            segment_end=milestone_emb.loc[mns["to"],],
                        )
                        tmp_progressions = mns.iloc[proj["segment"]-1][["from", "to"]]
                        tmp_progressions["cell_id"] = cids
                        tmp_progressions["percentage"] = proj["progression"]
                        tmp_progressions = tmp_progressions[["cell_id", "from", "to", "percentage"]].reset_index(drop=True)
                    else:
                        # self loop milestone
                        tmp_progressions = pd.DataFrame(data=[cell_id for cell_id in cids], columns=["cell_id"])
                        tmp_progressions["from"] = cluster
                        tmp_progressions["to"] = cluster
                        tmp_progressions["percentage"] = 1
                    progressions.append(tmp_progressions)
                else:
                    pass

            progressions = pd.concat(progressions)
            progressions.reset_index(drop=True)

        self.add_trajectory(
            milestone_network=milestone_network,
            divergence_regions=None,
            progressions=progressions,
        )

    def add_trajectory_graph(
            self,
            cell_graph: pd.DataFrame,
            to_keep: pd.Series | dict = None,
            milestone_prefix: str = "milestone_",
            backend: str = "networkx"
    ):
        if not "length" in cell_graph.columns:
            cell_graph["length"] = 1
        if not "directed" in cell_graph.columns:
            cell_graph["directed"] = False

        cell_ids = self.obs.index
        is_directed = cell_graph["directed"].any()

        # keep points are key cells for milestone network, where they have to appear.
        if to_keep is None:
            to_keep = pd.Series(True, index=cell_ids)
        elif type(to_keep) == dict:
            to_keep = pd.Series(to_keep)
        v_keeps = to_keep[to_keep].index.to_list()

        if backend.lower() == "networkx":
            # construct graph object using networkX as backend, which are more convenient for dataframe.
            G = nx.from_pandas_edgelist(cell_graph, source="from", target="to", edge_attr=["length", "directed"], create_using=nx.DiGraph if is_directed else nx.Graph)

            # simplify graph preliminary
            # step 1: for each cell, find closest milestone
            distance_df = pd.DataFrame(dict(nx.shortest_path_length(G.to_undirected(),  weight="length"))).loc[cell_ids, v_keeps]  # calucate distance as undirected graph, like "mode=all" in igraph
            closest_trajpoint = distance_df.idxmin(axis=1)  # closest keep point for each cell

            # step 2: simplify backbone
            G = G.subgraph(v_keeps)
            milestone_ids = G.nodes

            # STEP 3: Calculate progressions of cell_ids to determine which nodes were on each path
            milestone_network_proto = nx.to_pandas_edgelist(G, source="from", target="to")
            milestone_network_proto["path"] = milestone_network_proto.apply(lambda x: nx.shortest_path(G, source=x["from"], target=x["to"]), axis=1)
            # calculate progressions for keep point
            progressions_v_keeps = milestone_network_proto\
                .explode("path")\
                .groupby("path")\
                .agg(lambda x: x.iloc[0]).reset_index()\
                .rename(columns={"path": "node"})[["from", "to", "length", "node"]]  # save first edge for keep point
            progressions_v_keeps["percentage"] = progressions_v_keeps.apply(lambda x: nx.shortest_path_length(G, source=x["from"], target=x["node"],  weight="length")/x["length"], axis=1)

            closest_trajpoint_df = pd.DataFrame()
            closest_trajpoint_df["node"] = closest_trajpoint
            closest_trajpoint_df["cell_id"] = cell_ids
            progressions = pd.merge(progressions_v_keeps, closest_trajpoint_df, on="node")  # map all cells to closest keep point
            progressions = progressions[["cell_id", "from", "to", "percentage"]]

            milestone_network = milestone_network_proto[["from", "to", "length", "directed"]]

            # add prefix for milestone
            milestone_ids = [f"{milestone_prefix}{milestone_id}" for milestone_id in milestone_ids]
            milestone_network[["from", "to"]] = milestone_prefix + milestone_network[["from", "to"]]
            progressions[["from", "to"]] = milestone_prefix + progressions[["from", "to"]]
        else:
            # construct graph object using igraph as backend, which are faster
            milestone_network = None
            progressions = None

        # first add
        self.add_trajectory(
            milestone_network=milestone_network,
            divergence_regions=None,
            progressions=progressions
        )
        # simplify and add
        simplified_milestone_wrapper = self.simplify_trajectory(self.model_name)  # TODO: 此处轨迹简化有问题
        self.add_trajectory(
            milestone_network=simplified_milestone_wrapper["milestone_network"],
            divergence_regions=None,
            progressions=simplified_milestone_wrapper["progressions"]

        )

    # def add_trajectory_velocity(
    #         self,
    #         neighbors: dict,
    #         velocity: np.array,
    #         velocity_graph,
    #         cluster_key: str = None
    # ):
    #     # TODO: add velocity trajectory using PAGA transform, such as scVelo, VeloAE
    #     import scvelo as scv

    #     cluster_key = "clusters"
    #     shape = velocity.shape
    #     adata = self.copy()[:shape[0], :shape[1]] # 强行维度一致
    #     adata.uns["neighbors"] = neighbors
    #     adata.layers["velocity"] = velocity
    #     adata.uns["velocity_graph"] = velocity_graph

    #     scv.tl.paga(adata, groups=cluster_key)

    #     df = scv.get_df(adata, 'paga/transitions_confidence', precision=2).T

    #     milestone_network = df.reset_index()\
    #         .rename(columns={'index': 'from'})\
    #         .melt(id_vars="from", var_name="to", value_name="length")\
    #         .query("`length` > 0")
    #     milestone_network["length"] = 1  # 暂时统一设置为1
    #     milestone_network["directed"] = True

    #     obs = self.obs.reset_index()  # change index
    #     milestone_id_list = list(obs[cluster_key].cat.categories)
    #     X_emb = self.obsm["X_umap"]
    #     milestone_emb = np.array(list(obs.groupby(cluster_key).apply(lambda x: X_emb[list(x.index)].mean(axis=0))))
    #     milestone_emb = pd.DataFrame(milestone_emb, index=milestone_id_list)

    #     self.add_trajectory_projection(
    #         milestone_network=milestone_network,
    #         milestone_emb=milestone_emb,
    #         X_emb=X_emb,
    #         cluster_key=cluster_key
    #     )

    def group_onto_trajectory_edges(self, cluster_key="_cfe_te_group"):
        """group cells to edges
        ref: PyDynverse/pydynverse/wrap/wrap_add_grouping.group_onto_trajectory_edges

        Returns:
            pd.DataFrame: _description_
        """
        def get_trajectory_edges(x):
            x = x.loc[x["percentage"].idxmax()]
            return f"{x['from']}->{x['to']}"
        group_df = self.milestone_wrapper.progressions.groupby("cell_id").apply(get_trajectory_edges)
        self.obs[cluster_key] = group_df.loc[self.obs.index]

    def group_onto_nearest_milestones(self, cluster_key="_cfe_nm_group"):
        """ group cells to nearest milestones
        ref: PyDynverse/pydynverse/wrap/wrap_add_grouping.group_onto_nearest_milestones

        Returns:
            pd.DataFrame: _description_
        """

        def get_nearest_milestone(x):
            return x.loc[x["percentage"].idxmax(), "milestone_id"]
        group_df = self.milestone_wrapper.milestone_percentages.groupby("cell_id").apply(get_nearest_milestone)
        self.obs[cluster_key] = group_df.loc[self.obs.index]

    def simplify_trajectory(self, model_name="default") -> MilestoneWrapper:
        """ simplify trajectory for metric comparison, also used in FateAnnData.add_trajectory_cell_graph
        ref: PyDynverse/pydynverse/wrap/simplify_trajectory.py

        Args:
            model_name (_type_, optional): _description_. Defaults to None.

        Returns:
            MilestoneWrapper: simplified milestone_wrapper
        """
        if model_name in self.trajectory_history_dict:
            milestone_wrapper = self.trajectory_history_dict[model_name]["milestone_wrapper"]
        else:
            raise ValueError(f"model '{model_name}' not found in trajectory_history_dict")

        milestone_network = milestone_wrapper.milestone_network.copy()
        divergence_regions = milestone_wrapper.divergence_regions
        progressions = milestone_wrapper.progressions.copy()

        G = nx.from_pandas_edgelist(
            milestone_network.rename(columns={"length": "weight"}),  # need length to adjust weight
            source="from",
            target="to",
            edge_attr=True,
            create_using=nx.DiGraph if milestone_wrapper.directed else nx.Graph
        )

        # simplify cells
        edge_points = progressions
        edge_points.rename(columns={"cell_id": "id"}, inplace=True)
        edge_points["id"] = edge_points["id"].apply(lambda x: f"SIMPLIFYCELL_{x}")

        # core: simplify networkx network
        out = self._simplify_networkx_network(
            G,
            force_keep=divergence_regions["milestone_id"],
            edge_points=edge_points
        )

        # milestone data structure based on simplied network
        G = out["gr"]
        milestone_network = pd.DataFrame(G.edges(data=True), columns=["from", "to", "attributes"])
        milestone_network = pd.concat([milestone_network.drop(columns=['attributes']), milestone_network["attributes"].apply(pd.Series)], axis=1)
        milestone_network = milestone_network[["from", "to", "weight", "directed"]].rename(columns={"weight": "length"})

        edge_points = out["edge_points"]
        progressions = out["edge_points"][["id", "from", "to", "percentage"]].rename(columns={"id": "cell_id"})
        progressions["cell_id"] = progressions["cell_id"].apply(lambda x: x.replace("SIMPLIFYCELL_", ""))

        simplified_milestone_wrapper = MilestoneWrapper(
            milestone_network=milestone_network,
            divergence_regions=divergence_regions,
            progressions=progressions,
        )
        return simplified_milestone_wrapper

    def _simplify_networkx_network(self, G, force_keep, edge_points):
        # copy from: PyDynverse/pydynverse/wrap/simplify_networkx_network.py
        from ._simplify_networkx_network import simplify_networkx_network as snn
        return snn(G, force_keep=force_keep, edge_points=edge_points)

    def write_h5ad(self, save_cfe=True, *args, **kwargs):
        # if self.cfe_dict.get("milestone_wrapper", None) is not None:
        #     self.cfe_dict["milestone_wrapper"] = dict(self.cfe_dict["milestone_wrapper"])
        # if self.cfe_dict.get("waypoint_wrapper", None) is not None:
        #     self.cfe_dict["waypoint_wrapper"] = dict(self.cfe_dict["waypoint_wrapper"])
        #     self.cfe_dict["waypoint_wrapper"]["milestone_wrapper"] = None  # milestone_wrapper is redundent
        #     waypoints = self.cfe_dict["waypoint_wrapper"]["waypoints"]
        #     self.cfe_dict["waypoint_wrapper"]["waypoints"] = waypoints.fillna("")  # "" replace None
        if save_cfe:
            trajectory_history_dict = self.trajectory_history_dict
            for model_name, trajectory in trajectory_history_dict.items():
                if "milestone_wrapper" in trajectory:
                    milestone_wrapper = dict(trajectory["milestone_wrapper"])
                    self.trajectory_history_dict[model_name]["milestone_wrapper"] = milestone_wrapper
                if "waypoint_wrapper" in trajectory:
                    waypoint_wrapper = trajectory["waypoint_wrapper"]
                    waypoint_wrapper["milestone_wrapper"] = None  # milestone_wrapper is redundent
                    waypoint_wrapper["waypoints"] = waypoint_wrapper["waypoints"].fillna("")  # "" replace None
                    self.trajectory_history_dict[model_name]["waypoint_wrapper"] = milestone_wrapper

        return super().write_h5ad(*args, **kwargs)

    def __getitem__(self, key):
        sub_adata = super().__getitem__(key)
        sub_fadata = self.from_anndata(sub_adata)
        # TODO: add sub operation for all other attributes, such as prior_information, milestone_wrapper, wayppoint_wrapper, etc.
        return sub_fadata


def read_h5ad(*args, **kwargs):
    """_summary_

    Returns:
        _type_: _description_
    """
    # TODO: milestone_wrapper和waypoint_wrapper的读取添加，需要字典解析
    adata = sc.read_h5ad(*args, **kwargs)
    fadata = FateAnnData.from_anndata(adata)
    return fadata

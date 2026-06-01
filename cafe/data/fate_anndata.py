# import h5py
import os
import pickle

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

from .._logging import logger
from .._settings import settings
from ..util import random_time_string
from .fate_milestone_wrapper import MilestoneWrapper
from .fate_waypoint_wrapper import WaypointWrapper

# from anndata._io.specs.registry import _REGISTRY, IOSpec  # global I/O registry


class FateAnnData(ad.AnnData):
    """
    AnnData object for cafe (CelluAr Fate Explorer).

    Stores data related to cell fate exploration in the `object.uns["cafe"]` attribute.
    This class extends `anndata.AnnData` to provide specialized functionality for
    trajectory inference, visualization, and benchmarking.

    Attributes:
        cafe_dict (dict): A dictionary stored in `uns["cafe"]` containing all Cafe-specific data.
        id (str): A unique identifier for the FateAnnData object.
        prior_information (dict): Dictionary storing prior knowledge for trajectory inference (e.g., start cells, clusters).
        model_name (str): The name of the currently active trajectory model.
        trajectory_history_dict (dict): Dictionary storing results from different trajectory inference methods.
    """

    def __init__(self, name: str = "FateAnnData", *args, **kwargs):
        """Initialize the FateAnnData class.

        Args:
            name (str, optional): Name of the FateAnnData object. Defaults to "FateAnnData".
            *args: Variable length argument list passed to `anndata.AnnData`.
            **kwargs: Arbitrary keyword arguments passed to `anndata.AnnData`.
        """
        super().__init__(*args, **kwargs)

        # prior information is frequently used with common value in various method function
        # such as cluster_key, basis, start_cell
        self.recognize_prior_information()  # recognize prior information dict automatically

        # check result dir for method run result
        self.check_result_dir()

        self.embedding_cache = {}  # cache for basis/embedding data

    @property
    def id(self):
        if "id" not in self.uns:
            self.uns["id"] = random_time_string("FateAnnData")
        return self.uns["id"]

    @id.setter
    def id(self, value):
        self.uns["id"] = value

    @property
    def cafe_dict(self):
        if "cafe" not in self.uns:
            self.uns["cafe"] = {}
        return self.uns["cafe"]

    @cafe_dict.setter
    def cafe_dict(self, value):
        self.uns["cafe"] = value

    @property
    def prior_information(self):
        if "prior_information" not in self.cafe_dict:
            self.cafe_dict["prior_information"] = {}
        return self.cafe_dict["prior_information"]

    @prior_information.setter
    def prior_information(self, value):
        self.cafe_dict["prior_information"] = value

    @property
    def model_name(self):
        return self.cafe_dict.get("model_name", "default")

    @model_name.setter
    def model_name(self, value):
        self.cafe_dict["model_name"] = value

    @property
    def trajectory_history_dict(self):
        # trajectory_history_dict
        # ├── ref                                                       # ref trajectory
        # │   └── ...
        # └── scvelo (scvelo trajectory)                                # method name
        #     ├── milestone_wrapper → MilestoneWrapper object
        #     ├── waypoint_wrapper → WaypointWrapper object
        #     ├── raw_wrapper_dict                                      # for method result record
        #     │   ├── wrapper_type → str
        #     │   ├── ... other raw data
        #     ├── trajectory_embedding → dict                           # for visualization
        #     │   └── X_umap → dict                                     # embedding basis
        #     │       ├── wp_segments → DataFrame shape=(210, 9)
        #     │       └── milestone_positions → DataFrame shape=(14, 9)
        #     └── resource_usage → dict                                 # for benchmark
        if "trajectory_history_dict" not in self.cafe_dict:
            self.cafe_dict["trajectory_history_dict"] = {}
        return self.cafe_dict["trajectory_history_dict"]

    @trajectory_history_dict.setter
    def trajectory_history_dict(self, value):
        self.cafe_dict["trajectory_history_dict"] = value

    @property
    def milestone_wrapper(self):
        # return self._milestone_wrapper
        # model_dict = self.trajectory_history_dict.get(self.model_name, None)
        # if model_dict is not None:
        #     return model_dict.get("milestone_wrapper")
        # else:
        #     return None
        return self.trajectory_history_dict.get(self.model_name, {}).get("milestone_wrapper", None)

    @milestone_wrapper.setter
    def milestone_wrapper(self, value):
        # self._milestone_wrapper = value
        model_dict = self.trajectory_history_dict.get(self.model_name, None)
        if model_dict is not None:
            model_dict["milestone_wrapper"] = value
        else:
            self.trajectory_history_dict[self.model_name] = {"milestone_wrapper": value}

    @property
    def waypoint_wrapper(self):
        # return self._waypoint_wrapper
        # model_dict = self.trajectory_history_dict.get(self.model_name, None)
        # if model_dict is not None:
        #     return model_dict.get("waypoint_wrapper")
        # else:
        #     return None
        return self.trajectory_history_dict.get(self.model_name, {}).get("waypoint_wrapper", None)

    @waypoint_wrapper.setter
    def waypoint_wrapper(self, value):
        # self._waypoint_wrapper = value
        model_dict = self.trajectory_history_dict.get(self.model_name, None)
        if model_dict is not None:
            model_dict["waypoint_wrapper"] = value
        else:
            self.trajectory_history_dict[self.model_name] = {"waypoint_wrapper": value}

    @property
    def raw_wrapper_dict(self):
        return self.trajectory_history_dict.get(self.model_name, {}).get("raw_wrapper_dict", {})

    @raw_wrapper_dict.setter
    def raw_wrapper_dict(self, value):
        model_dict = self.trajectory_history_dict.get(self.model_name, None)
        if model_dict is not None:
            model_dict["raw_wrapper_dict"] = value
        else:
            self.trajectory_history_dict[self.model_name] = {"raw_wrapper_dict": value}

    @property
    def wrapper_type(self):
        return self.trajectory_history_dict.get(self.model_name, {}).get("wrapper_type", {})

    @wrapper_type.setter
    def wrapper_type(self, value):
        self.cafe_dict["wrapper_type"] = value

    # the readonly property
    @property
    def is_wrapped_with_trajectory(self):
        return "milestone_wrapper" in self.trajectory_history_dict.get(self.model_name, {})

    @property
    def is_wrapped_with_waypoints(self):
        return "waypoint_wrapper" in self.trajectory_history_dict.get(self.model_name, {})

    # these above functions are properties for single trajectory management
    # these following function: get_xxx and set_xxx methods can be used for multi-trajectory management
    def get_trajectory_dict(self, model_name: str = None):
        model_name = self.parse_model_name(model_name)
        if model_name is None:
            return None
        else:
            trajectory_dict = self.trajectory_history_dict[model_name]
            return trajectory_dict

    def set_trajectory_dict(self, trajectory_dict: dict, model_name=None):
        if model_name is None:
            model_name = self.model_name
        self.trajectory_history_dict[model_name] = trajectory_dict

    def get_milestone_wrapper(self, model_name=None):
        model_name = self.parse_model_name(model_name)
        return self.get_trajectory_dict(model_name)["milestone_wrapper"]

    def set_milestone_wrapper(self, milestone_wrapper: MilestoneWrapper, model_name=None):
        self.get_trajectory_dict(model_name)["milestone_wrapper"] = milestone_wrapper

    def get_waypoint_wrapper(self, model_name=None):
        model_name = self.parse_model_name(model_name)
        trajectory_dict = self.get_trajectory_dict(model_name)
        if "waypoint_wrapper" not in trajectory_dict:
            logger.warning(f"waypoint_wrapper not found in trajectory_dict for model '{model_name}'")
            return None
        else:
            return trajectory_dict["waypoint_wrapper"]

    def set_waypoint_wrapper(self, waypoint_wrapper: WaypointWrapper, model_name=None):
        self.get_trajectory_dict(model_name)["waypoint_wrapper"] = waypoint_wrapper

    def get_raw_wrapper_dict(self, model_name=None):
        model_name = self.parse_model_name(model_name)
        trajectory_dict = self.get_trajectory_dict(model_name)
        if "raw_wrapper_dict" not in trajectory_dict:
            logger.warning(f"raw_wrapper_dict not found in trajectory_dict for model '{model_name}'")
            return None
        else:
            return trajectory_dict["raw_wrapper_dict"]

    def parse_model_name(self, model_name: str = None):
        model_name_list = self.get_all_model_name(parse=False)
        if model_name is None:
            model_name = self.model_name
        elif model_name in model_name_list:
            pass
        else:
            # try match the parsed and raw trajectory ID
            parsed_model_name_list = self.get_all_model_name(parse=True)
            parsed2raw = dict(zip(parsed_model_name_list, model_name_list))
            if model_name in parsed2raw.keys():
                raw_model_name = parsed2raw[model_name]
                logger.debug(f"match pased:'{model_name}' to raw:'{raw_model_name}'")
                model_name = raw_model_name

        if model_name not in self.trajectory_history_dict:
            logger.debug(f"model '{model_name}' not found in trajectory_history_dict")
            return None
        return model_name

    @classmethod
    def from_anndata(cls, adata: ad.AnnData) -> "FateAnnData":
        """Create a FateAnnData object from an existing AnnData object.

        Args:
            adata (ad.AnnData): existing AnnData object

        Returns:
            fadata (cafe.data.FateAnnData): generated FateAnnData object
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
            obsp=adata.obsp,
            layers=adata.layers,
        )

        return fadata

    def to_anndata(self, delete_trajectory=False):
        uns = self.uns.copy()
        if delete_trajectory and ("cafe" in uns):
            del uns["cafe"]
        adata = ad.AnnData(
            X=self.X,
            obs=self.obs,
            var=self.var,
            uns=uns,
            obsm=self.obsm,
            varm=self.varm,
            obsp=self.obsp,
            layers=self.layers,
        )
        return adata

    def add_prior_information(self, **kwargs) -> None:
        """Add prior information to the FateAnnData object.

        ref: pydynverse/wrap/wrap_add_prior_information add_prior_information
        """
        self.prior_information.update(kwargs)

    def recognize_prior_information(self):
        # recognize prior information dict automatically

        logger.debug("recognizing prior information...")
        prior_information = {}
        # cluster and basis are chosen by candidate list priority.
        cluster_candidate_list = ["clusters", "celltype"]
        basis_candidate_list = ["X_umap", "X_tsne", "X_pca", "X_emb"]
        for cluster_candidate in cluster_candidate_list:
            if cluster_candidate in self.obs.columns:
                prior_information["cluster"] = cluster_candidate
                logger.debug(f"recognize '{cluster_candidate}' in '.obs' columns as 'cluster' key", indent_level=2)
                break
        for basis_candidate in basis_candidate_list:
            if basis_candidate in self.obsm.keys():
                prior_information["basis"] = basis_candidate
                logger.debug(f"recognize '{basis_candidate}' in '.obsm' keys as 'basis' key", indent_level=2)
                break
        # TODO: start_cell need specified
        self.prior_information.update(prior_information)

    def get_prior_infomation_dynverse():
        # get prior information with dynverse style
        return

    def add_model_name(self, model_name: str):
        self.model_name = model_name
        # self.cafe_dict["model_name"] = model_name
        self.trajectory_history_dict[self.model_name] = {}

    def get_parsed_model_name(self, model_name: str = None):
        from ..util import parse_random_time_string

        if model_name is None:
            model_name = self.model_name
        return parse_random_time_string(model_name)

    def get_all_model_name(self, parse=True):
        model_name_list = list(self.trajectory_history_dict.keys())
        if self.model_name not in self.trajectory_history_dict:
            model_name_list = [self.model_name] + model_name_list
        if parse:
            model_name_list = [self.get_parsed_model_name(i) for i in model_name_list]
        return model_name_list

    def add_resource_usage(self, resource_usage: dict) -> None:
        """Add resource usage to the FateAnnData object.

        Args:
            resource_usage (dict): resource usage dict, such as {"time": 26.1, "memory": 845320, "cpu": 0.99,}
        """
        if self.model_name not in self.trajectory_history_dict:
            self.trajectory_history_dict[self.model_name] = {}
        self.get_trajectory_dict(self.model_name)["resource_usage"] = resource_usage

    def get_resource_usage(self, model_name: str = None) -> dict:
        """Get resource usage for a specific model."""
        if model_name is None:
            model_name = self.model_name
        return self.get_trajectory_dict(model_name).get("resource_usage", {})

    # def get_all_resource_usage(self):
    #     """Get resource usage for all models."""
    #     resource_usage_dict = {}
    #     for model_name in self.trajectory_history_dict:
    #         resource_usage_dict[model_name] = self.get_resource_usage(model_name)
    #     return resource_usage_dict

    def add_trajectory(
        self,
        milestone_network: pd.DataFrame,
        milestone_id_list: list = None,
        divergence_regions: pd.DataFrame = None,
        milestone_percentages: pd.DataFrame = None,
        progressions: pd.DataFrame = None,
        generate_color: bool = True,
        wrapper_type: str = "direct",
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
            milestone_id_list=milestone_id_list,
            cell_id_list=None,  # may lose cells, should extract from milestone_percentages["cell_id"]
            divergence_regions=divergence_regions,
            milestone_percentages=milestone_percentages,
            progressions=progressions,
            wrapper_type=wrapper_type,
        )
        # synchronize mielstone color with cluster color in prior_information if possible
        if generate_color:
            cluster = self.prior_information.get("cluster")
            if cluster and (f"{cluster}_colors" in self.uns):
                ref_color_dict = dict(zip(self.obs[cluster].cat.categories.tolist(), self.uns[f"{cluster}_colors"]))
            else:
                ref_color_dict = None
            milestone_wrapper._generate_color(ref_color_dict=ref_color_dict)

        self.milestone_wrapper = milestone_wrapper

        # save multiple trajectory in cafe_dict
        if self.model_name not in self.trajectory_history_dict:
            self.trajectory_history_dict[self.model_name] = {}
        self.trajectory_history_dict[self.model_name]["milestone_wrapper"] = milestone_wrapper
        # trajectory wrapper raw data, which is different for linear, projection, graph and etc.
        self.trajectory_history_dict[self.model_name]["raw_wrapper_dict"] = self.raw_wrapper_dict
        self.trajectory_history_dict[self.model_name]["trajectory_embedding"] = {}

    def add_trajectory_mannually(
        self,
        milestone_network: pd.DataFrame,
        wrapper_type: str = "projection",
        cluster: str = None,
        basis: str = "X_umap",
        distance_metric: str = "euclidean",
        model_name: str = "ref",
    ):
        """add trajectory mannually as ref trajectory, reuse add_trajectory_projection to get progression

        Args:
            milestone_network (pd.DataFrame): milestone network
            wrapper_type (str, optional): trajectory wrapper type, can be "projection" or "cluster".
            cluster (str, optional): cluster key for cluster.
            basis (str, optional): cell embedding key.
            distance_metric (str, optional): distance metric.
            model_name (str, optional): trajectory model name.
        """
        if cluster is None:
            cluster = self.prior_information.get("cluster", "clusters")
        self.add_model_name(model_name)

        if wrapper_type == "projection":
            from sklearn.metrics.pairwise import pairwise_distances

            obs = self.obs.reset_index()  # change index
            milestone_id_list = list(obs[cluster].cat.categories)
            X_emb = self.obsm[basis]
            milestone_emb = np.array(list(obs.groupby(cluster).apply(lambda x: X_emb[list(x.index)].mean(axis=0))))
            milestone_emb = pd.DataFrame(milestone_emb, index=milestone_id_list)
            # self.obs = self.obs.set_index("index")

            # milestone network
            dis = pd.DataFrame(
                pairwise_distances(milestone_emb, metric=distance_metric),
                index=milestone_id_list,
                columns=milestone_id_list,
            )
            milestone_network["length"] = milestone_network.apply(lambda row: dis.loc[row["from"], row["to"]], axis=1)
            milestone_network["directed"] = True

            # progressions
            self.wrapper_type = "projection"
            self.add_trajectory_projection(milestone_network=milestone_network, milestone_emb=milestone_emb, X_emb=X_emb, cluster_key=cluster)
        elif wrapper_type == "cluster":
            if "length" not in milestone_network.columns:
                milestone_network["length"] = 1
            if "directed" not in milestone_network.columns:
                milestone_network["directed"] = True
            self.wrapper_type = "cluster"
            self.add_trajectory_cluster(
                milestone_network=milestone_network,
                cluster=cluster,
            )

        else:
            raise Exception(f"parameter wrapper_type '{wrapper_type}' not supported in add_trajectory_mannually")

    def add_trajectory_by_type(self, trajectory_dict: dict, **kwargs) -> None:
        """automatically add trajectory by wrapper type in trajectory_dict

        Args:
            trajectory_dict (dict): _description_
        """
        wrapper_type = trajectory_dict["wrapper_type"]
        self.wrapper_type = wrapper_type
        logger.debug(f"Add trajectory by wrapper type: {wrapper_type}")
        self.raw_wrapper_dict = trajectory_dict

        if wrapper_type == "directed":
            self.add_trajectory(**trajectory_dict, **kwargs)
        elif wrapper_type == "branch":
            self.add_trajectory_branch(
                branch_network=trajectory_dict["branch_network"],
                branches=trajectory_dict["branches"],
                branch_progressions=trajectory_dict["branch_progressions"],
                **kwargs,
            )
        elif wrapper_type == "linear":
            self.add_trajectory_linear(pseudotime=trajectory_dict["pseudotime"], **kwargs)
        elif wrapper_type == "cycle":
            self.add_trajectory_cycle(pseudotime=trajectory_dict["pseudotime"], **kwargs)
        elif wrapper_type == "probability":
            self.add_trajectory_probability(
                end_state_probabilities=trajectory_dict["end_state_probabilities"],
                pseudotime=trajectory_dict["pseudotime"] if "pseudotime" in trajectory_dict.keys() else None,
                **kwargs,
            )
        elif wrapper_type == "cluster":
            self.add_trajectory_cluster(milestone_network=trajectory_dict["milestone_network"], cluster=trajectory_dict["cluster"], **kwargs)
        elif wrapper_type == "projection":
            self.add_trajectory_projection(
                milestone_network=trajectory_dict["milestone_network"],
                milestone_emb=trajectory_dict["milestone_emb"],
                X_emb=trajectory_dict["X_emb"],
                cluster_key=trajectory_dict.get("cluster_key", None),
                **kwargs,
            )
        elif wrapper_type == "graph":
            self.add_trajectory_graph(cell_graph=trajectory_dict["cell_graph"], to_keep=trajectory_dict["to_keep"], **kwargs)
        elif wrapper_type == "velocity":
            self.add_trajectory_velocity(
                velocity=trajectory_dict["velocity"],
                velocity_graph=trajectory_dict.get("velocity_graph"),
                velocity_graph_neg=trajectory_dict.get("velocity_graph_neg"),
                velocity_embedding=trajectory_dict.get("velocity_embedding"),
                neighbors=trajectory_dict.get("neighbors"),
                obs_index=trajectory_dict.get("obs_index"),
                var_index=trajectory_dict.get("var_index"),
                X=trajectory_dict.get("X"),  # add X for velocity method like veloae,
                **kwargs,
            )
        elif wrapper_type == "lineage":
            # TODO: fix lineage trajectory for cellrank
            self.add_trajectory_lineage(
                probability=trajectory_dict["probability"],
                cluster_key=trajectory_dict.get("cluster_key", None),
                new_cluster_list=trajectory_dict.get("new_cluster_list", None),
                **kwargs,
            )
        elif wrapper_type == "time":
            self.add_trajectory_time(
                tmaps=trajectory_dict["tmaps"],
                time_key=trajectory_dict.get("time_key", None),
                cluster_key=trajectory_dict.get("cluster_key", None),
                flow_threshold=trajectory_dict.get("flow_threshold", 0.1),
                relative_threshold=trajectory_dict.get("relative_threshold", 0.3),
                normalize=trajectory_dict.get("normalize", True),
                include_self_loop=trajectory_dict.get("include_self_loop", False),
            )

    def add_waypoints(self, milestone_wrapper: MilestoneWrapper = None, model_name: str = None, waypoint_wrapper_kwargs: dict = {}) -> None:
        """Create WaypointWrapper object"""
        logger.debug("FateAnnData add_waypoints")

        milestone_wrapper = (
            milestone_wrapper if milestone_wrapper is not None else self.get_milestone_wrapper(model_name)
        )  # waypoint is based on milestone
        waypoint_wrapper = WaypointWrapper(milestone_wrapper, **waypoint_wrapper_kwargs)
        # waypoint_wrapper.waypoint_geodesic_distances = waypoint_wrapper.waypoint_geodesic_distances.loc[:,self.obs.index] #
        # self.waypoint_wrapper = waypoint_wrapper
        # self.cafe_dict["waypoint_wrapper"] = waypoint_wrapper
        # self.is_wrapped_with_waypoints = True

        # if model_name not in self.trajectory_history_dict:
        #     self.trajectory_history_dict[model_name] = {}
        # self.trajectory_history_dict[model_name]["waypoint_wrapper"] = waypoint_wrapper
        self.set_waypoint_wrapper(waypoint_wrapper, model_name)

    def subset_trajectory(
        self,
        edge_list: list,
        model_name: str = None,
        cluster: str = None,
        keep_color_cluster: str = None,
    ) -> "FateAnnData":
        """
        Subset the FateAnnData object based on trajectory edges.

        Args:
            edge_list (list): list of edge tuples [('from', 'to'), ...]
            model_name (str): model name to subset. Defaults to current model.
        """
        if model_name is None:
            model_name = self.model_name

        mw = self.get_milestone_wrapper(model_name)
        new_mw = mw.subset_by_edges(edge_list)  # milestone keep here

        # subset adata
        new_fadata = self[new_mw.cell_id_list].copy()
        new_fadata.id = f"{self.id}_subset_{edge_list}"
        new_fadata.check_result_dir()

        # keep cell and milestone color with raw fadata if possible
        if keep_color_cluster is not None:
            cluster = keep_color_cluster
        else:
            cluster = self.prior_information.get("cluster")
        if set(new_fadata.obs[cluster].unique()) == set(new_mw.id_list):
            new_fadata.obs[cluster] = pd.Categorical(new_fadata.obs[cluster], categories=new_mw.id_list)  # ensure the category order
            new_fadata.uns[f"{cluster}_colors"] = [new_mw.milestone_color_dict[k] for k in new_mw.id_list]

        # update the wrapper in the new object
        new_fadata.set_milestone_wrapper(new_mw, model_name=model_name)

        # Remove waypoint wrapper for this model as it might be invalid now
        # Or ideally, re-initialize it?
        # For safety, let's remove it from the history of new_fadata
        traj_dict = new_fadata.get_trajectory_dict(model_name)
        if "waypoint_wrapper" in traj_dict:
            del traj_dict["waypoint_wrapper"]
            new_fadata.is_wrapped_with_waypoints = False

        return new_fadata

    # TODO: core operation should move to MilestoneWrapper, the interface should refer to the next function
    def merge_edge_trajectory(self, fadata_sub: "FateAnnData", replace_edges: list = None, model_name: str = None):
        """
        Merge a fine-grained trajectory (from fadata_sub) back into the coarse trajectory (self).

        Args:
            fadata_sub (FateAnnData): The subset FateAnnData object containing the fine-grained trajectory.
            replace_edges (list): List of edges [('from', 'to')] in the current trajectory to be removed and replaced.
            model_name (str): The model name to update. Defaults to current model.
        """
        if model_name is None:
            model_name = self.model_name

        global_mw = self.get_milestone_wrapper(model_name)
        # Assuming fadata_sub uses its own default model
        local_mw = fadata_sub.get_milestone_wrapper()

        if local_mw is None:
            raise ValueError("fadata_sub does not have a valid MilestoneWrapper.")

        # 1. Merge Milestone Network
        # Remove replaced edges from global
        new_mn = global_mw.milestone_network.copy()
        if replace_edges:
            for u, v in replace_edges:
                # remove rows where from=u and to=v
                # Use boolean indexing for deletion
                mask = (new_mn["from"] == u) & (new_mn["to"] == v)
                new_mn = new_mn[~mask]

        # Add local edges
        local_mn = local_mw.milestone_network.copy()
        new_mn = pd.concat([new_mn, local_mn], ignore_index=True).drop_duplicates()

        # 2. Merge Progressions
        sub_cell_ids = fadata_sub.obs_names
        global_prog = global_mw.progressions

        # Keep global progressions for cells NOT in sub
        keep_mask = ~global_prog["cell_id"].isin(sub_cell_ids)
        new_prog = global_prog[keep_mask].copy()

        # Add local progressions
        local_prog = local_mw.progressions.copy()
        new_prog = pd.concat([new_prog, local_prog], ignore_index=True)

        # 3. Create new MilestoneWrapper and update
        # We reuse the add_trajectory machinery to handle wrapper creation and registration
        self.add_trajectory(
            milestone_network=new_mn,
            progressions=new_prog,
            # Let divergence_regions be re-calculated or lost if not maintained manually.
            # Ideally we should merge them if present.
            divergence_regions=None,
            generate_color=False,  # Don't overwrite colors if not necessary, maybe?
        )
        # TODO: scale the edge length in new_mn if needed, to maintain consistency with global trajectory

        logger.info(f"Successfully merged edge trajectory from subset with {len(fadata_sub)} cells.")
        return self

    # TODO
    # def merge_edge_trajectory(
    #     self,
    #     fadata_sub: "FateAnnData",
    #     replace_edge: str,
    #     model_name: str = None,
    #     sub_model_name: str = None,
    #     save_model_name: str = "merge",
    #     scale_local_edge_length: bool = True,
    # ):
    #     target_model_name = self.parse_model_name(model_name)
    #     if target_model_name is None:
    #         raise ValueError(f"model '{model_name}' not found in current trajectory history")
    #     target_sub_model_name = fadata_sub.parse_model_name(sub_model_name)
    #     if target_sub_model_name is None:
    #         raise ValueError(f"sub model '{sub_model_name}' not found in fadata_sub trajectory history")
    #     missing_cell_list = list(set(fadata_sub.obs.index) - set(self.obs.index))
    #     if len(missing_cell_list) > 0:
    #         raise ValueError(f"fadata_sub has {len(missing_cell_list)} cells not in self: {missing_cell_list}")

    #     mw = self.get_milestone_wrapper(target_model_name)
    #     sub_mw = fadata_sub.get_milestone_wrapper(target_sub_model_name)
    #     new_mw = mw.merge_edge_trajectory(
    #         sub_mw=sub_mw,
    #         replace_edge=replace_edge,
    #         scale_local_edge_length=scale_local_edge_length,
    #     )

    #     self.add_model_name(save_model_name)
    #     self.add_trajectory(
    #         milestone_network=new_mw.milestone_network,
    #         progressions=new_mw.progressions,
    #         divergence_regions=new_mw.divergence_regions,
    #         generate_color=False,
    #     )
    #     return self

    def merge_milestone_trajectory(
        self,
        fadata_sub: "FateAnnData",
        replace_milestone: str,
        model_name: str = None,
        sub_model_name: str = None,
        save_model_name: str = "merge",
        scale_local_edge_length: bool = True,
    ):
        target_model_name = self.parse_model_name(model_name)
        if target_model_name is None:
            raise ValueError(f"model '{model_name}' not found in current trajectory history")
        target_sub_model_name = fadata_sub.parse_model_name(sub_model_name)
        if target_sub_model_name is None:
            raise ValueError(f"sub model '{sub_model_name}' not found in fadata_sub trajectory history")

        missing_cell_list = list(set(fadata_sub.obs.index) - set(self.obs.index))
        if len(missing_cell_list) > 0:
            raise ValueError(f"fadata_sub has {len(missing_cell_list)} cells not in self: {missing_cell_list}")

        mw = self.get_milestone_wrapper(target_model_name)
        sub_mw = fadata_sub.get_milestone_wrapper(target_sub_model_name)
        new_mw = mw.merge_milestone_trajectory(
            sub_mw=sub_mw,
            replace_milestone=replace_milestone,
            scale_local_edge_length=scale_local_edge_length,
        )
        self.add_model_name(save_model_name)
        self.add_trajectory(
            milestone_network=new_mw.milestone_network,
            progressions=new_mw.progressions,
            divergence_regions=new_mw.divergence_regions,
            generate_color=False,
        )
        return self

    # fix

    def __getitem__(self, index):
        # 1. call Anndata __getitem__ to get the sliced AnnData object
        new_adata = super().__getitem__(index)

        # 2. directly set it to FateAnndata
        new_adata.__class__ = FateAnnData

        # Decouple uns so that cafe_dict property writes don't affect parent
        # We want to preserve other uns data, but isolate cafe data.
        new_adata.uns = self.uns.copy()
        if "cafe" in new_adata.uns:
            new_adata.uns["cafe"] = new_adata.uns["cafe"].copy()
        else:
            new_adata.uns["cafe"] = {}

        # 3. copy simple attribute/property from 'self' to 'new_adata'
        new_adata.prior_information = self.prior_information  # TODO: check
        new_adata.model_name = self.model_name
        # new_adata.id = f"{self.id}_subset"  # update id for subset
        # new_adata.check_result_dir()

        # 4. link complex trajectory attribute from 'self' to 'new_adata'
        # New trajectory history dict construction
        new_trajectory_history_dict = {}
        for model_name, trajectory_history in self.trajectory_history_dict.items():
            # Create copy to avoid modifying parent dict
            th_copy = trajectory_history.copy()

            if "milestone_wrapper" in th_copy:
                mw = th_copy["milestone_wrapper"]
                new_mw = mw.subset_by_cells(new_adata.obs_names.tolist())
                th_copy["milestone_wrapper"] = new_mw

            if "waypoint_wrapper" in th_copy:
                del th_copy["waypoint_wrapper"]  # directly remove waypoint wrapper for safety

            new_trajectory_history_dict[model_name] = th_copy

        new_adata.trajectory_history_dict = new_trajectory_history_dict
        new_adata.embedding_cache = {}

        return new_adata

    def copy(self, filename: str = None) -> "FateAnnData":
        """
        Full copy, optionally of some elements only.
        """
        # 1. Create a standard AnnData copy (this deep copies .uns)
        new_adata = super().copy(filename)

        # 2. Cast to FateAnnData
        if not isinstance(new_adata, FateAnnData):
            new_adata.__class__ = FateAnnData

        # related properties are stored in the self.uns["cafe"] attribute. So no need to copy again.
        return new_adata
        # # 3. Initialize FateAnnData specific attributes
        # new_adata.id = self.id

        # # NOTE: cafe_dict and its derived properties (prior_information, etc.)
        # # are automatically available via properties reading from new_adata.uns['cafe']

        # # Copy other auxiliary attributes that might not be in uns
        # # raw_wrapper_dict can be mutable, so we copy it
        # new_adata.raw_wrapper_dict = self.raw_wrapper_dict.copy() if self.raw_wrapper_dict else {}
        # new_adata.wrapper_type = self.wrapper_type
        # new_adata.is_wrapped_with_trajectory = self.is_wrapped_with_trajectory
        # new_adata.is_wrapped_with_waypoints = self.is_wrapped_with_waypoints

        # # embedding_cache is transient, copy it
        # new_adata.embedding_cache = self.embedding_cache.copy()

        # return new_adata
        # # #  deep copy milestone_wrapper and waypoint_wrapper if exist
        # # #  filter cells in milestone_wrapper and waypoint_wrapper if exist
        # # new_adata.uns["cafe"] = self.uns["cafe"]
        # # new_adata.cafe_dict = self.cafe_dict
        # # new_adata.trajectory_history_dict = self.trajectory_history_dict

        # # return new_adata

    def add_trajectory_branch(self, branch_network: pd.DataFrame, branch_progressions: pd.DataFrame, branches: pd.DataFrame) -> None:
        """Add branch trajectory,such as PAGA

        ref: PyDynverse/pydynverse/wrap/wrap_add_branch_trajectory.add_branch_trajectory

        Args:
            branch_network (pd.DataFrame): branch network with column list: ["from", "to"]
            branch_progressions (pd.DataFrame): branch progressions with column list: ["cell_id", "branch_id", "percentage"
            branches (pd.DataFrame): branches with column list: ["branch_id", "length", "directed"]
        """
        logger.debug("FateAnnData add_trajectory_branch")

        branch_id_list = branches["branch_id"]
        milestone_network = pd.DataFrame(
            {
                "from": map(lambda x: f"{x}_from", branch_id_list),
                "to": map(lambda x: f"{x}_to", branch_id_list),
                "branch_id": branch_id_list,
            }
        )
        milestone_mapper_network = pd.concat(
            [
                # single from node
                pd.DataFrame(
                    {
                        "from": map(lambda x: f"{x}_from", branch_id_list),
                        "to": map(lambda x: f"{x}_from", branch_id_list),
                    }
                ),
                # connected node, if "A->B" in branch_network , then "A_to->B_from" in here,
                pd.DataFrame(
                    {
                        "from": map(lambda x: f"{x}_to", branch_network["from"]),
                        "to": map(lambda x: f"{x}_from", branch_network["to"]),
                    }
                ),
                # single to node
                pd.DataFrame(
                    {
                        "from": map(lambda x: f"{x}_to", branch_id_list),
                        "to": map(lambda x: f"{x}_to", branch_id_list),
                    }
                ),
            ]
        )
        # transform node name to connected component id
        mapper = {}
        graph = nx.from_pandas_edgelist(milestone_mapper_network, source="from", target="to")
        connected_components = nx.connected_components(graph)
        for component_index, component in enumerate(connected_components):
            for node in component:
                # milestone id starts from 1
                mapper[node] = str(component_index + 1)
        milestone_network["from"] = milestone_network["from"].apply(lambda x: mapper[x])
        milestone_network["to"] = milestone_network["to"].apply(lambda x: mapper[x])
        milestone_network = pd.merge(milestone_network, branches, on="branch_id")

        progressions = pd.merge(branch_progressions, milestone_network, on="branch_id")[["cell_id", "from", "to", "percentage"]]

        milestone_network = milestone_network[["from", "to", "length", "directed"]]

        self.add_trajectory(milestone_network=milestone_network, progressions=progressions)

    def add_trajectory_linear(
        self,
        pseudotime: list,
        directed: bool = True,
        do_scale_minmax: bool = True,
    ) -> None:
        """add linear trajectory, such as Comp1(baseline), Palantir(TODO), Cytotrace(TODO).

        ref: PyDynverse/pydynverse/wrap/wrap_add_linear_trajector.add_linear_trajectory

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
        milestone_network = pd.DataFrame(
            {
                "from": milestone_ids[0],
                "to": milestone_ids[1],
                "length": 1,
                "directed": directed,
            },
            index=[0],
        )  # all scalar, need "index" to show sample num
        # progressions datafram construction， percentage=pseudotime
        progressions = pd.DataFrame(
            {
                "cell_id": self.obs.index,
                "from": milestone_ids[0],
                "to": milestone_ids[1],
                "percentage": pseudotime,
            }
        )
        self.add_trajectory(
            milestone_network=milestone_network,
            divergence_regions=None,
            progressions=progressions,
            wrapper_type="linear",
        )

    def add_trajectory_cycle(
        self,
        pseudotime: list,
        directed: bool = False,
        do_scale_minmax: bool = True,
    ) -> None:
        """add cycle trajectory, such as Angle(baseline).
        ref: PyDynverse/pydynverse/wrap/wrap_add_cyclic_trajectory.add_cyclic_trajectory

        Args:
            pseudotime (list): pseudotime sequence.
            directed (bool, optional): is directed graph. Defaults to False.
            do_scale_minmax (bool, optional): scale pseudotime to [0, 1]. Defaults to True.
        """
        pseudotime = np.array(pseudotime)

        # min-max scale pseudotime to [0, 1]
        if do_scale_minmax:
            pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
        else:
            assert (pseudotime >= 0).all() and (pseudotime <= 1).all()

        # milestone_network: A->B, B->C, C->A
        milestone_ids = ["A", "B", "C"]
        milestone_network = pd.DataFrame(
            {
                "from": milestone_ids,
                "to": milestone_ids[1:] + [milestone_ids[0]],
                "length": 1,
                "directed": directed,
                "edge_id": range(len(milestone_ids)),
            }
        )

        # progression: 3 segement
        progressions = pd.DataFrame(
            {
                "cell_id": self.obs.index,
                "time": [3 * i for i in pseudotime],
            }
        )
        progressions["edge_id"] = progressions["time"].apply(lambda x: 0 if x <= 1 else 1 if x <= 2 else 2).astype("int")
        progressions = pd.merge(progressions, milestone_network[["from", "to", "edge_id"]], on="edge_id")
        progressions["percentage"] = progressions["time"] - progressions["edge_id"]
        progressions = progressions[["cell_id", "from", "to", "percentage"]].reset_index(drop=True)

        milestone_network = milestone_network[["from", "to", "length", "directed"]]

        self.add_trajectory(
            milestone_network=milestone_network,
            divergence_regions=None,
            progressions=progressions,
            wrapper_type="cycle",
        )

    def add_trajectory_probability(self, end_state_probabilities: pd.DataFrame, pseudotime: list = None, do_scale_minmax: bool = True):
        """add probability trajectory, such as StatComp(baseline), Palantir.

        ref: PyDynverse/pydynverse/wrap/wrap_add_end_state_probabilities.add_end_state_probabilities

        Args:
            end_state_probabilities (pd.DataFrame): the probability from start point to multiple endpoint.
            pseudotime (list): pseudotime sequence
            do_scale_minmax (bool, optional): scale pseudotime to [0, 1]. Defaults to True.
        """
        # TODO: optimize this strategy to new wrapper: lineage.

        if pseudotime is None:
            pseudotime = np.ones(end_state_probabilities.shape[0])
            do_scale_minmax = False
        if do_scale_minmax:
            pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())

        if end_state_probabilities.shape[1] == 1:
            # there is only one terminal state, which is a linear trajectory
            self.add_trajectory_linear(
                pseudotime=pseudotime,
                directed=True,
                do_scale_minmax=do_scale_minmax,
            )
        else:
            # multiple terminal states, building a milestone network
            # the starting point is a completely virtual point
            start_milestone_id = "milestone_begin"
            # the terminal point is extracted from the column name, and the default first column is cell_id
            if "cell_id" not in end_state_probabilities.columns:
                end_state_probabilities["cell_id"] = self.obs.index.tolist()
            end_milestone_ids = end_state_probabilities.columns.tolist()
            end_milestone_ids.remove("cell_id")
            milestone_ids = [start_milestone_id] + end_milestone_ids

            # star shaped milestone network with starting point as the center
            milestone_network = pd.DataFrame({"from": start_milestone_id, "to": end_milestone_ids, "length": 1, "directed": True})

            # add a divergence region composed of all milestone nodes together
            divergence_regions = pd.DataFrame(
                {
                    "milestone_id": milestone_ids,
                    "divergence_id": "D",
                    "is_start": pd.Series(milestone_ids) == start_milestone_id,
                }
            )

            pseudotime = pd.Series(pseudotime, index=end_state_probabilities["cell_id"])
            progressions = end_state_probabilities.melt(id_vars=["cell_id"], var_name="to", value_name="percentage")
            progressions["from"] = start_milestone_id
            progressions["percentage"] = progressions.groupby("cell_id")["percentage"].transform(
                lambda x: x / x.sum() * pseudotime[x.name]
            )  # 缩放使其之和为1，暂时不理解这个
            progressions = progressions[["cell_id", "from", "to", "percentage"]]

            self.add_trajectory(
                milestone_network=milestone_network,
                divergence_regions=divergence_regions,
                progressions=progressions,
                wrapper_type="probability",
            )

    def add_trajectory_cluster(
        self,
        milestone_network: pd.DataFrame,
        cluster: str | list,
        add_direction: bool = False,
    ):
        """add cluster trajectory, such as ClusterMST(baseline).

        ref: PyDynverse/pydynverse/wrap/wrap_add_cluster_graph.add_cluster_graph

        Args:
            milestone_network (pd.DataFrame): milestone network.
            cluster (str | list): cluster key or list.
        """
        # if add_direction:
        #     # TODO: fix for undirected graph
        #     logger.debug("try to add direction for undirected graph use prior information: 'start_milestone' or 'start_cell'")

        if isinstance(cluster, str):
            cluster_list = self.obs[cluster]
        else:
            cluster_list = pd.Series(cluster, index=self.obs.index)
        mn_ft = milestone_network[["from", "to"]]
        both_direction = pd.concat([mn_ft.assign(label=mn_ft["from"], percentage=0), mn_ft.assign(label=mn_ft["to"], percentage=1)])

        # TODO: fix for alone milestone 'stavia'
        progressions = (
            pd.DataFrame({"cell_id": self.obs.index, "label": cluster_list})
            .merge(both_direction, on="label")
            .groupby("cell_id")
            .apply(lambda x: x.sort_values("percentage", ascending=False).iloc[0])
            .reset_index(drop=True)
            .drop("label", axis=1)
        )

        self.add_trajectory(
            milestone_network=milestone_network,
            divergence_regions=None,
            progressions=progressions,
            wrapper_type="cluster",
        )

    def add_trajectory_projection(
        self,
        milestone_network: pd.DataFrame,
        milestone_emb: pd.DataFrame,
        X_emb: pd.DataFrame | np.ndarray | str,
        cluster_key: str = None,
    ):
        """add projection trajectory, such as CellMST(baseline).

        ref: PyDynverse/pydynverse/wrap/wrap_add_dimred_projection.add_dimred_projection

        Args:
            milestone_network (pd.DataFrame): milestone network.
            milestone_emb (pd.DataFrame): embbeding for milestones.
            X_emb (pd.DataFrame | np.ndarray | str): embedding for cells.
            cluster_key (str, optional): cluster key.
        """
        from ..util import project_to_segments

        if isinstance(X_emb, str):
            X_emb = self.obsm[X_emb]
            cell_id_list = self.obs.index.tolist()
        elif isinstance(X_emb, pd.DataFrame):
            if X_emb.index.dtype == int:
                # for method cluster mst, reset index from int to cell_id
                X_emb.index = self.obs.iloc[X_emb.index].index
            cell_id_list = self.obs.loc[X_emb.index].index.tolist()  # intersection of cell id
            if len(cell_id_list) < self.shape[0]:
                cell_lost_list = set(self.obs.index) - set(cell_id_list)
                logger.warning(f"cell lost during trajectory projection: {cell_lost_list}")
        else:
            # ndarray
            cell_id_list = self.obs.index.tolist()
            X_emb = pd.DataFrame(X_emb, index=cell_id_list)

        # add self loop for discrete isolated milestone
        discrete_milestones = list(set(milestone_emb.index) - (set(milestone_network["from"]) | set(milestone_network["to"])))
        if len(discrete_milestones) > 0:
            logger.info(f"discrete milestones: {discrete_milestones}")
            self_loop_milestone_network = pd.DataFrame()
            self_loop_milestone_network["from"] = discrete_milestones
            self_loop_milestone_network["to"] = discrete_milestones
            self_loop_milestone_network["length"] = 0
            self_loop_milestone_network["directed"] = False
            milestone_network = milestone_network.append(self_loop_milestone_network)

        if cluster_key is None:
            # if no cluster key is given, just project all cells to the segments
            proj = project_to_segments(
                x=X_emb,
                segment_start=milestone_emb.loc[milestone_network["from"],],
                segment_end=milestone_emb.loc[milestone_network["to"],],
            )
            progressions = milestone_network.iloc[proj["segment"] - 1][["from", "to"]]
            progressions["cell_id"] = X_emb.index
            progressions["percentage"] = proj["progression"]
            progressions = progressions[["cell_id", "from", "to", "percentage"]].reset_index(drop=True)
        else:
            # project cells onto the line segments corresponding to their respective clusters
            cluster_series = self[X_emb.index.tolist()].obs[cluster_key]
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
                        tmp_progressions = mns.iloc[proj["segment"] - 1][["from", "to"]]
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
            milestone_id_list=milestone_emb.index.tolist(),
            divergence_regions=None,
            progressions=progressions,
            wrapper_type="projection",
        )

    def add_trajectory_graph(
        self,
        cell_graph: pd.DataFrame,
        to_keep: pd.Series | dict = None,
        milestone_prefix: str = "milestone_",
        backend: str = "networkx",
        simplify_kwargs: dict = {},
    ):
        """add graph trajectory, such as GraphMST(baseline).

        ref: PyDynverse/pydynverse/wrap/wrap_add_cell_graph.add_cell_graph

        Args:
            cell_graph (pd.DataFrame): _description_
            to_keep (pd.Series | dict, optional): _description_. Defaults to None.
            milestone_prefix (str, optional): _description_. Defaults to "milestone_".
            backend (str, optional): _description_. Defaults to "networkx".
        """
        if "length" not in cell_graph.columns:
            cell_graph["length"] = 1
        if "directed" not in cell_graph.columns:
            cell_graph["directed"] = False

        if "prune_threshold" not in simplify_kwargs:
            # for dataset 'pancreas' and method 'Graph MST' , threnshold is best
            simplify_kwargs["prune_threshold"] = 0.05

        is_directed = cell_graph["directed"].any()
        cell_ids = list(pd.unique(pd.concat([cell_graph["from"], cell_graph["to"]])))
        if len(cell_ids) < self.shape[0]:
            cell_lost_list = set(self.obs.index) - set(cell_ids)
            logger.warning(f"cell lost during trajectory graph construction: {cell_lost_list}")

        # keep points are key cells for milestone network, where they have to appear.
        if to_keep is None:
            to_keep = pd.Series(True, index=cell_ids)
        elif isinstance(to_keep, dict):
            to_keep = pd.Series(to_keep)
        v_keeps = to_keep[to_keep].index.to_list()

        if backend.lower() == "networkx":
            # construct graph object using networkX as backend, which are more convenient for dataframe.
            G = nx.from_pandas_edgelist(
                cell_graph,
                source="from",
                target="to",
                edge_attr=["length", "directed"],
                create_using=nx.DiGraph if is_directed else nx.Graph,
            )

            # simplify graph preliminary
            # step 1: for each cell, find closest milestone
            # calucate distance as undirected graph, like "mode=all" in igraph
            distance_df = pd.DataFrame(dict(nx.shortest_path_length(G.to_undirected(), weight="length")))
            distance_df = distance_df.loc[cell_ids, v_keeps]
            closest_trajpoint = distance_df.idxmin(axis=1)  # closest keep point for each cell

            # step 2: simplify backbone
            G = G.subgraph(v_keeps)
            milestone_ids = G.nodes

            # STEP 3: Calculate progressions of cell_ids to determine which nodes were on each path
            milestone_network_proto = nx.to_pandas_edgelist(G, source="from", target="to")
            milestone_network_proto["path"] = milestone_network_proto.apply(lambda x: nx.shortest_path(G, source=x["from"], target=x["to"]), axis=1)
            # calculate progressions for keep point
            progressions_v_keeps = (
                milestone_network_proto.explode("path")
                .groupby("path")
                .agg(lambda x: x.iloc[0])
                .reset_index()
                .rename(columns={"path": "node"})[["from", "to", "length", "node"]]
            )  # save first edge for keep point
            progressions_v_keeps["percentage"] = progressions_v_keeps.apply(
                lambda x: nx.shortest_path_length(G, source=x["from"], target=x["node"], weight="length") / x["length"],
                axis=1,
            )

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
            # TODO: construct graph object using igraph as backend, which are faster
            milestone_network = None
            progressions = None

        # first add
        self.add_trajectory(
            milestone_network=milestone_network,
            divergence_regions=None,
            progressions=progressions,
            generate_color=False,  # here there are many milestone, don't generate color
        )
        # simplify and add
        simplified_milestone_wrapper = self.simplify_trajectory(self.model_name, simplify_kwargs=simplify_kwargs)  # TODO: update
        # TODO: new lost cells
        self.add_trajectory(
            milestone_network=simplified_milestone_wrapper["milestone_network"],
            divergence_regions=None,
            progressions=simplified_milestone_wrapper["progressions"],
            wrapper_type="graph",
        )

    def add_trajectory_lineage(
        self,
        probability: pd.DataFrame,
        cluster_key: str = None,
        new_cluster_list: list = None,
        strategy: str = "graph_fusion",  # base, graph_fusion, hierarchical_clustering
        **strategy_kwargs,
    ):
        # TODO: for palantir, cellrank
        from ._lineage_wrapper import LINEAGE_STRATEGIES

        logger.debug(f"Adding lineage trajectory using '{strategy}' strategy...")

        strategy_func = LINEAGE_STRATEGIES[strategy]
        trajectory_components = strategy_func(
            fadata=self, probability=probability, cluster_key=cluster_key, new_cluster_list=new_cluster_list, **strategy_kwargs
        )

        if trajectory_components is None:
            logger.warning(f"Failed to add lineage trajectory using '{strategy}' strategy.")
        else:
            self.add_trajectory(
                milestone_network=trajectory_components["milestone_network"],
                divergence_regions=trajectory_components.get("divergence_regions"),
                progressions=trajectory_components["progressions"],
                wrapper_type="lineage",
            )
            logger.debug(f"Successfully added lineage trajectory using '{strategy}' strategy.")

    # TODO: Time wrapper for WaddingtonOT, Moscot
    # TODO:
    def add_trajectory_time(
        self,
        tmaps: dict,
        time_key: str = None,
        cluster_key: str = None,
        flow_threshold: float = 0.1,
        relative_threshold: float = 0.3,
        normalize: bool = True,
        include_self_loop: bool = False,
    ):
        """Add trajectory from time-series optimal transport results (WaddingtonOT, Moscot).

        This method aggregates cell-level transport matrices into cluster-level transitions,
        then constructs milestone_network and progressions for cafe trajectory.

        Edge selection strategy (both conditions must be met):
        1. Absolute threshold: flow > flow_threshold
        2. Relative threshold: flow > relative_threshold * max_outgoing_flow

        This allows preserving bifurcations while filtering out noise edges.

        Args:
            tmaps: dict, keys are (t_start, t_end) tuples, values are transport matrices
                   of shape (n_cells_t_start, n_cells_t_end) representing transition probabilities.
            time_key: str, column name in obs for time points. If None, uses prior_information.
            cluster_key: str, column name in obs for cell clusters. If None, uses prior_information.
            flow_threshold: float, absolute minimum flow to include an edge (default 0.1).
            relative_threshold: float, keep edges with flow >= relative_threshold * max_flow (default 0.3).
                               Set to 0 to disable relative filtering.
            normalize: bool, whether to normalize transition matrix by row.
            include_self_loop: bool, whether to include self-loop edges (A->A).

        Example:
            >>> fadata.add_trajectory_time(
            ...     tmaps=tmaps_moscot,
            ...     time_key="time",
            ...     cluster_key="celltype",
            ...     flow_threshold=0.1,      # 绝对阈值：过滤噪声
            ...     relative_threshold=0.3,  # 相对阈值：保留 ≥30% 最大流量的边
            ... )
        """
        from scipy import sparse

        logger.debug("FateAnnData add_trajectory_time")

        # Get keys from prior_information if not specified
        if time_key is None:
            time_key = self.prior_information.get("time_key", "time")
        if cluster_key is None:
            cluster_key = self.prior_information.get("cluster", "clusters")

        obs = self.obs
        clusters = list(obs[cluster_key].cat.categories)
        n_clusters = len(clusters)
        cluster_to_idx = {c: i for i, c in enumerate(clusters)}

        # ========== Step 1: Build cluster indicator matrices (for matrix multiplication) ==========
        def build_indicator_matrix(time_val):
            """Build sparse indicator matrix G_t (n_cells_t x n_clusters)"""
            mask = obs[time_key] == time_val
            cell_indices = np.where(mask.values)[0]
            cluster_codes = obs.loc[mask, cluster_key].map(cluster_to_idx).values
            n_cells = len(cell_indices)
            data = np.ones(n_cells, dtype=float)
            G = sparse.csr_matrix((data, (np.arange(n_cells), cluster_codes)), shape=(n_cells, n_clusters))
            return G

        # ========== Step 2: Aggregate cell-level Tmaps to cluster-level flow ==========
        cluster_flow = np.zeros((n_clusters, n_clusters))

        logger.debug(f"Aggregating {len(tmaps)} time-pair transport matrices...")
        for (t1, t2), tmap in tmaps.items():
            # Validate dimensions
            n_c1 = (obs[time_key] == t1).sum()
            n_c2 = (obs[time_key] == t2).sum()
            if tmap.shape != (n_c1, n_c2):
                logger.warning(f"Skipping {t1}->{t2}: Tmap shape {tmap.shape} != expected ({n_c1}, {n_c2})")
                continue

            # Build indicator matrices
            G1 = build_indicator_matrix(t1)
            G2 = build_indicator_matrix(t2)

            # Matrix multiplication: ClusterFlow = G1.T @ Tmap @ G2
            if sparse.issparse(tmap):
                flow = G1.T @ tmap @ G2
            else:
                flow = G1.T @ sparse.csr_matrix(tmap) @ G2
            cluster_flow += flow.toarray() if sparse.issparse(flow) else flow

        # Normalize by row
        if normalize:
            row_sums = cluster_flow.sum(axis=1, keepdims=True)
            cluster_flow = cluster_flow / (row_sums + 1e-10)

        cluster_flow_df = pd.DataFrame(cluster_flow, index=clusters, columns=clusters)

        # ========== Step 3: Build milestone_network from cluster flow ==========
        # Strategy: Use both absolute and relative thresholds to preserve bifurcations
        edges = []
        for source in clusters:
            outgoing = cluster_flow_df.loc[source].copy()

            # Optionally exclude self-loop
            if not include_self_loop:
                outgoing = outgoing.drop(source, errors="ignore")

            if len(outgoing) == 0 or outgoing.max() == 0:
                # No valid outgoing edges, add self-loop as fallback
                edges.append(
                    {
                        "from": source,
                        "to": source,
                        "length": 1.0,
                        "directed": True,
                        "flow": cluster_flow_df.loc[source, source] if source in cluster_flow_df.columns else 0,
                    }
                )
                continue

            # Compute dynamic threshold based on max flow
            max_flow = outgoing.max()
            dynamic_threshold = max(flow_threshold, relative_threshold * max_flow)

            # Filter edges by combined threshold
            valid_targets = outgoing[outgoing >= dynamic_threshold]

            if len(valid_targets) == 0:
                # Fallback: keep the strongest edge
                valid_targets = outgoing.nlargest(1)

            for target, flow in valid_targets.items():
                edges.append(
                    {
                        "from": source,
                        "to": target,
                        "length": 1.0 / (flow + 1e-6),  # Higher flow → shorter length
                        "directed": True,
                        "flow": flow,
                    }
                )

        if not edges:
            logger.warning("No edges found above flow_threshold. Consider lowering the threshold.")
            # Add self-loops as fallback
            for c in clusters:
                edges.append({"from": c, "to": c, "length": 1.0, "directed": True, "flow": 1.0})

        milestone_network = pd.DataFrame(edges)

        # ========== Step 4: Build progressions (assign cells to edges) ==========
        # Strategy: Assign each cell to the edge (source_cluster -> target_cluster)
        # where source_cluster is the cell's cluster, and target_cluster is chosen
        # based on the maximum outgoing flow. Percentage is based on time position.

        time_values = obs[time_key].cat.categories.tolist()
        time_to_norm = {t: i / max(len(time_values) - 1, 1) for i, t in enumerate(time_values)}

        progressions_list = []
        for cell_id in obs.index:
            cell_cluster = obs.loc[cell_id, cluster_key]
            cell_time = obs.loc[cell_id, time_key]

            # Find the best target cluster (highest flow from this cluster)
            outgoing = cluster_flow_df.loc[cell_cluster]
            # Exclude self-loop if there are other options
            if (outgoing.drop(cell_cluster, errors="ignore") > flow_threshold).any():
                target_cluster = outgoing.drop(cell_cluster, errors="ignore").idxmax()
            else:
                target_cluster = cell_cluster  # Self-loop

            # Percentage based on normalized time
            percentage = time_to_norm.get(cell_time, 0.5)

            progressions_list.append(
                {
                    "cell_id": cell_id,
                    "from": cell_cluster,
                    "to": target_cluster,
                    "percentage": percentage,
                }
            )

        progressions = pd.DataFrame(progressions_list)

        # ========== Step 5: Call add_trajectory ==========
        self.add_trajectory(
            milestone_network=milestone_network[["from", "to", "length", "directed"]],
            progressions=progressions,
        )

        # Store additional info in raw_wrapper_dict
        self.raw_wrapper_dict["cluster_flow"] = cluster_flow_df
        self.raw_wrapper_dict["tmaps_keys"] = list(tmaps.keys())

        logger.debug(f"Added time trajectory with {len(milestone_network)} edges and {len(progressions)} cell progressions.")

    def add_trajectory_velocity(
        self,
        velocity: np.array,
        velocity_graph: np.array,
        velocity_graph_neg: np.array,
        velocity_embedding: np.array,
        neighbors: dict,
        milestone_network_strategy: str = "paga",
        cluster: str = None,
        obs_index=None,
        var_index=None,
        basis=None,
        X: np.array = None,
    ):
        # TODO: move to _velocity_wrapper module
        "add velocity trajectory using PAGA transform, such as scVelo, VeloAE"
        if cluster is None:
            cluster = self.prior_information.get("cluster")
        if basis is None:
            basis = self.prior_information.get("basis")

        # PAGA
        import scvelo as scv

        if X is not None:
            # for veloae
            adata = ad.AnnData(X)
            adata.obs.index = obs_index if obs_index is not None else self.obs.index
            adata.var.index = var_index if var_index is not None else self.var.index
            adata.obs[cluster] = self[adata.obs.index].obs[cluster]
            adata.obsm[basis] = self[adata.obs.index].obsm[basis]
        else:
            # extract sub adata
            if (obs_index is not None) or (var_index is not None):
                obs_index = self.obs.index if obs_index is None else obs_index
                var_index = self.var.index if var_index is None else var_index
                adata = self[obs_index, var_index].copy()
            else:
                # TODO: copy may waste time and memory, need other strategy
                # adata = self.copy()
                adata = self.to_anndata()

        logger.debug(f"filterd adata: {adata}")

        velocity_basis = f"velocity_{basis[2:]}"
        if velocity_embedding is not None:
            milestone_network_strategy = "low_dim_paga"  # force to use cons strategy
            logger.debug(f"use given velocity embedding, use strategy '{milestone_network_strategy}' to get milestone_network")
        else:
            adata.layers["velocity"] = velocity
            if (velocity_graph is not None) and (velocity_graph_neg is not None):
                # Final goal: only save velocity matrix of a method.
                adata.uns["velocity_graph"] = velocity_graph
                adata.uns["velocity_graph_neg"] = velocity_graph_neg
                adata.uns["neighbors"] = {}
                adata.obsp["distances"] = neighbors["distances"]
                adata.obsp["connectivities"] = neighbors["connectivities"]
            else:
                # recompute neighbors and velocity graph may waste time
                scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
                scv.tl.velocity_graph(adata)  # add transition graph by velocity

            logger.debug("add raw velocity embedding to fadata")
            scv.tl.velocity_embedding(adata, basis=basis[2:])
            velocity_embedding = adata.obsm[velocity_basis]
        self.raw_wrapper_dict.update({velocity_basis: velocity_embedding})

        # compute milestone embedding based clustered cell embedding
        X_emb = pd.DataFrame(adata.obsm[basis], index=adata.obs.index)
        milestone_emb = adata.obs.groupby(cluster).apply(lambda x: X_emb.loc[x.index].mean(axis=0))
        milestone_emb.index = list(adata.obs[cluster].cat.categories)

        # construct milestone_network based velocity
        if milestone_network_strategy == "paga":
            # use paga based graph connectivity
            scv.tl.paga(adata, groups=cluster)
            df = scv.get_df(adata, "paga/transitions_confidence", precision=2).T
            # df.index = df.columns = adata.obs[cluster].cat.categories.tolist()
            milestone_network = (
                df.reset_index().rename(columns={"index": "from"}).melt(id_vars="from", var_name="to", value_name="length").query("`length` > 0")
            )
            milestone_network["length"] = 1  # TODO: need to be modified based embedding distance between milestone.
            milestone_network["directed"] = True
        elif milestone_network_strategy == "low_dim_paga":
            # paga based on expression embedding and velocity embedding
            new_adata = sc.AnnData(X=adata.obsm[basis], obs=adata.obs, obsm=adata.obsm, obsp=adata.obsp, uns=adata.uns)
            new_adata.layers["spliced"] = adata.obsm[basis]
            new_adata.layers["unspliced"] = adata.obsm[basis]
            new_adata.layers["velocity"] = velocity_embedding
            # recomput velocity graph based on low-dim velocity and embedding
            sc.pp.neighbors(new_adata)
            scv.tl.velocity_graph(new_adata, show_progress_bar=False)
            scv.tl.paga(new_adata, groups=cluster)  # recompute paga
            df = scv.get_df(adata, "paga/transitions_confidence", precision=2).T
            print(df)
            # df.index = df.columns = adata.obs[cluster].cat.categories.tolist()
            milestone_network = (
                df.reset_index().rename(columns={"index": "from"}).melt(id_vars="from", var_name="to", value_name="length").query("`length` > 0")
            )
            milestone_network["length"] = 1  # TODO: need to be modified based embedding distance between milestone.
            milestone_network["directed"] = True
        else:
            # TODO: use velocity consine similarity method, need fix
            threshold = 0.2
            cluster_list = adata.obs[cluster].cat.categories.to_list()
            cluster_connection_df = pd.DataFrame(0.0, index=cluster_list, columns=cluster_list)
            for source_cluster in cluster_list:
                source_cell_velocity = velocity_embedding[np.where(self.obs[cluster] == source_cluster)[0]]
                source_cell_velocity = source_cell_velocity / (np.linalg.norm(source_cell_velocity, axis=1, keepdims=True) + 1e-6)
                for target_cluster in cluster_list:
                    if source_cluster == target_cluster:
                        continue
                    cluster_velocity = milestone_emb.loc[target_cluster].values - milestone_emb.loc[source_cluster].values
                    cluster_velocity = cluster_velocity / (np.linalg.norm(cluster_velocity) + 1e-6)
                    # cosine similarity between each cell's velocity and the inter-cluster direction
                    # normalized vector dot calculation is equal to cosin similarity calculation.
                    cosine_sims = (source_cell_velocity @ cluster_velocity).mean()
                    # TODO: weighted
                    cluster_connection_df.loc[source_cluster, target_cluster] = cosine_sims
            logger.debug(f"cluster_connection_df:\n{cluster_connection_df.round(2)}")
            milestone_network = cluster_connection_df.stack().reset_index()
            milestone_network.columns = ["from", "to", "score"]
            milestone_network = milestone_network[milestone_network["score"] > threshold].copy()
            milestone_network["length"] = 1.0
            milestone_network["directed"] = True
        # TODO: other strategy LAP

        X_emb = pd.DataFrame(self.obsm[basis], index=self.obs.index)  # use all cell
        self.add_trajectory_projection(milestone_network=milestone_network, milestone_emb=milestone_emb, X_emb=X_emb, cluster_key=cluster)

    def add_metric(
        self,
        metric_dict: dict,
        model_name: str = None,
    ):
        if model_name is None:
            model_name = self.model_name
        self.trajectory_history_dict[model_name]["metric_dict"] = metric_dict

    def get_metric(self):
        pass

    def group_onto_trajectory_edges(self, model_name=None, cluster_key="_cafe_te_group"):
        """group cells to edges
        ref: PyDynverse/pydynverse/wrap/wrap_add_grouping.group_onto_trajectory_edges

        Returns:
            pd.DataFrame: _description_
        """

        def get_trajectory_edges(x):
            x = x.loc[x["percentage"].idxmax()]
            return f"{x['from']}->{x['to']}"

        mw = self.get_trajectory_dict(model_name)["milestone_wrapper"]
        group_df = mw.progressions.groupby("cell_id").apply(get_trajectory_edges)
        self.obs[cluster_key] = None
        self.obs.loc[group_df.index, cluster_key] = group_df

    def group_onto_nearest_milestones(self, model_name=None, cluster_key="_cafe_nm_group"):
        """group cells to nearest milestones
        ref: PyDynverse/pydynverse/wrap/wrap_add_grouping.group_onto_nearest_milestones

        Returns:
            pd.DataFrame: _description_
        """

        # don't modify MilestoneWrapper object, only get obs attribute
        # mw.group_onto_nearest_milestones get new MilestoneWrapper object
        def get_nearest_milestone(x):
            return x.loc[x["percentage"].idxmax(), "milestone_id"]

        mw = self.get_trajectory_dict(model_name)["milestone_wrapper"]
        group_df = mw.milestone_percentages.groupby("cell_id").apply(get_nearest_milestone)

        self.obs[cluster_key] = None
        self.obs.loc[group_df.index, cluster_key] = group_df

    def simplify_trajectory(self, model_name="default", simplify_kwargs: dict = {}) -> MilestoneWrapper:
        """simplify trajectory for metric comparison, also used in FateAnnData.add_trajectory_cell_graph
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
            # need length to adjust weight
            milestone_network.rename(columns={"length": "weight"}),
            source="from",
            target="to",
            edge_attr=True,
            create_using=nx.DiGraph if milestone_wrapper.directed else nx.Graph,
        )

        # simplify cells
        edge_points = progressions
        edge_points.rename(columns={"cell_id": "id"}, inplace=True)
        edge_points["id"] = edge_points["id"].apply(lambda x: f"SIMPLIFYCELL_{x}")

        # core: simplify networkx network
        from ._simplify_networkx_network import simplify_networkx_network as snn

        out = snn(G, force_keep=divergence_regions["milestone_id"], edge_points=edge_points, **simplify_kwargs)

        # milestone data structure based on simplied network
        G = out["gr"]
        milestone_network = pd.DataFrame(G.edges(data=True), columns=["from", "to", "attributes"])
        milestone_network = pd.concat([milestone_network.drop(columns=["attributes"]), milestone_network["attributes"].apply(pd.Series)], axis=1)
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

    def get_trajectory_embedding(self, basis=None, model_name=None):
        if model_name is None:
            model_name = self.model_name
        if basis is None:
            basis = self.prior_information.get("basis")
        trajectory_embedding = self.get_trajectory_dict(model_name)["trajectory_embedding"]
        return trajectory_embedding.get(basis, None)

    def set_trajectory_embedding(self, wp_segments, milestone_positions, basis=None, model_name=None):
        if model_name is None:
            model_name = self.model_name
        if basis is None:
            basis = self.prior_information.get("basis")
        self.get_trajectory_dict(model_name)["trajectory_embedding"][basis] = {
            "wp_segments": wp_segments.replace({None: ""}),
            "milestone_positions": milestone_positions,
        }

    def get_start_milestone(self, start_cell, model_name=None):
        # get start milestone based on start cell, find the milestone with highest percentage for the cell
        trajectory_dict = self.get_trajectory_dict(model_name)

        milestone_wrapper = trajectory_dict["milestone_wrapper"]
        milestone_percentages = milestone_wrapper.milestone_percentages
        start_cell_percentages = milestone_percentages.query(f"cell_id == '{start_cell}'")

        if start_cell_percentages.shape[0] == 0:
            raise Exception(f"start cell '{start_cell}' is not available")

        # find the max milestone percentage of the cell as start milestone
        max_idx = start_cell_percentages["percentage"].idxmax()
        start_milestone = start_cell_percentages.loc[max_idx]["milestone_id"]

        return start_milestone

    def _check_start_milestone(self, start_milestone=None, start_cell=None, model_name=None):
        trajectory_dict = self.get_trajectory_dict(model_name)

        start_milestone = start_milestone if start_milestone else self.prior_information.get("start_milestone")

        # use_start_cell = False
        # if start_milestone is None:
        #     logger.debug(f"start_milestone is None, try to use start cell('{start_cell}') to identify start milestone automatically")
        #     use_start_cell = True
        # elif start_milestone not in trajectory_dict["milestone_wrapper"].id_list:
        #     logger.debug(
        #         f"start_milestone '{start_milestone}' not in milestone list, try to use start cell('{start_cell}') to identify start milestone automatically"
        #     )
        #     use_start_cell = True

        if (start_milestone is None) or (start_milestone not in trajectory_dict["milestone_wrapper"].id_list):
            use_start_cell = True
        else:
            use_start_cell = False

        if use_start_cell:
            logger.debug("try to use start cell to identify start milestone automatically")
            start_cell = start_cell if start_cell else self.prior_information.get("start_cell")
            if start_cell is None:
                raise Exception("start_milestone and start_cell are both None")
            else:
                start_milestone = self.get_start_milestone(start_cell, model_name=model_name)
            logger.debug(f"find start milestone '{start_milestone}' from start cell '{start_cell}'")

        return start_milestone

    def get_trajectory_pseudotime(self, start_milestone=None, start_cell=None, model_name=None):
        # get trajectory pseudotime based on start_milestone

        start_milestone = self._check_start_milestone(start_milestone=start_milestone, start_cell=start_cell, model_name=model_name)
        trajectory_dict = self.get_trajectory_dict(model_name)

        pseudotime_key = f"pseudotime_from_{start_milestone}"
        if pseudotime_key in trajectory_dict:
            # return pseudotime from trajectory dict directly
            logger.debug(f"find key:'{pseudotime_key}' in trajectory dict, use it directly")
            return trajectory_dict[pseudotime_key]
        else:
            # calculate new pseudotime
            logger.debug("calculating new pseudotime")
            milestone_wrapper = trajectory_dict["milestone_wrapper"]
            # claculate the distance from the starting milestone to each milestone
            milestone_network = milestone_wrapper.milestone_network
            is_directed = milestone_network["directed"].any()
            G = nx.from_pandas_edgelist(
                milestone_network,
                source="from",
                target="to",
                edge_attr=["length"],
                create_using=nx.DiGraph if is_directed else nx.Graph,
            )
            m_spl_dict = nx.shortest_path_length(G, source=start_milestone, weight="length")
            unconnected_milestone_list = list(set(G.nodes) - set(m_spl_dict.keys()))
            if unconnected_milestone_list:
                logger.warning(f"unconnected milestones found: {unconnected_milestone_list}")
                m_spl_dict.update({i: None for i in unconnected_milestone_list})  # fix for milestone that is not connected to start_milestone
            m_spl_df = pd.DataFrame.from_dict(m_spl_dict, orient="index", columns=["distance"])

            # calculate cell distance from start milestone,
            def calculate_cell_pseudotime(cell_group):
                distances = m_spl_df.loc[cell_group["milestone_id"], "distance"]
                if distances.isnull().any():
                    return np.nan
                percentages = cell_group["percentage"].values
                return (distances * percentages).sum()

            milestone_percentages = milestone_wrapper.milestone_percentages
            pseudotime = milestone_percentages.groupby("cell_id").apply(calculate_cell_pseudotime).loc[self.obs.index]
            # set unconnected cell pseudotime to random value between 0 and 1
            nan_mask = pseudotime.isnull()
            num_nans = nan_mask.sum()
            if num_nans > 0:
                logger.debug(f"Filling {num_nans} NaN pseudotime values with random numbers between 0 and 1.")
                random_values = np.random.rand(num_nans)
                pseudotime.loc[nan_mask] = random_values

            # save pseudotime
            logger.debug(f"save pseudotime to trajectory dict with key: `{pseudotime_key}`")
            trajectory_dict[pseudotime_key] = pseudotime.tolist()
            return pseudotime

    def get_trajectory_pseudo_velocity(self, basis=None, model_name=None):
        # TODO: another strategy, consider about waypoint

        # 1,2 calc milestone positions in embedding space: refer to cafe.plot.project_waypoints
        # 1. extract trajectory and cell embedding
        milestone_wrapper = self.get_milestone_wrapper(model_name)
        if basis is None:
            basis = self.prior_information.get("basis")
        cell_embedding = self.obsm[basis]
        cell_embedding = pd.DataFrame(cell_embedding, index=self.obs.index)

        milestone_network = milestone_wrapper.milestone_network
        progressions = milestone_wrapper.progressions
        milestone_percentages = milestone_wrapper.milestone_percentages

        # 2. merge and calc weighted avg milestone embedding
        merged_df = milestone_percentages.merge(cell_embedding, left_on="cell_id", right_index=True)

        def weighted_avg(group):
            coords = group.iloc[:, -cell_embedding.shape[1] :]
            weights = group["percentage"]
            # if weights.sum() == 0:
            #     return pd.Series(np.nan, index=coords.columns)
            return (coords.multiply(weights, axis=0)).sum() / weights.sum()

        milestone_embedding = merged_df.groupby("milestone_id").apply(weighted_avg)

        # 3. calc pseudovelocity vectors for each cell
        edge_vectors = milestone_embedding.loc[milestone_network["to"]].values - milestone_embedding.loc[milestone_network["from"]].values
        edge_vectors_df = pd.DataFrame(edge_vectors, index=pd.MultiIndex.from_frame(milestone_network[["from", "to"]]))

        # Map each cell's progression to its corresponding edge vector
        prog_with_vectors = progressions.join(edge_vectors_df, on=["from", "to"])
        prog_with_vectors.fillna(0, inplace=True)  # for cells on milestone, velocity = 0

        def weighted_avg_velocity(group):
            # For each cell, calculate the weighted average of its associated edge vectors
            # Extract vectors and weights
            vectors = group.iloc[:, -cell_embedding.shape[1] :].values
            weights = group["percentage"].values
            # Calculate weighted average: sum(vector * weight) / sum(weights)
            weighted_vectors = vectors * weights[:, np.newaxis]
            sum_of_weights = weights.sum()

            if sum_of_weights > 0:
                return weighted_vectors.sum(axis=0) / sum_of_weights
            else:
                # Return a zero vector if weights sum to 0 to avoid division by zero
                return np.zeros(cell_embedding.shape[1])

        # Group by cell_id and apply the weighted average calculation
        velocity_df = prog_with_vectors.groupby("cell_id").apply(weighted_avg_velocity)
        velocity_df = pd.DataFrame(velocity_df.to_list(), index=velocity_df.index)
        velocity_df = velocity_df.loc[self.obs.index]
        velocity_embedding = velocity_df.values
        return velocity_embedding

    def get_lineages(self, start_milestone=None, start_cell=None, target_milestone_list=None, model_name=None, return_element_type="obs_index"):
        # TODO: DFS from root to find all lineage for downstream driver gene search
        # ref: notebook_dev/hzy/downstream_lineage_dev.ipynb
        # 1. check start milestone, target milestone list
        start_milestone = self._check_start_milestone(start_milestone=start_milestone, start_cell=start_cell, model_name=model_name)

        mw = self.get_milestone_wrapper(model_name)
        G = mw.milestone_network_G

        # available target milestone is the leaf node milestone in the same subgraph with start_milestone
        available_target_milestone_list = [node for node in G.nodes if nx.has_path(G, start_milestone, node) and G.out_degree(node) == 0]
        if target_milestone_list is None:
            target_milestone_list = available_target_milestone_list
        else:
            # check target_milstone
            invalid_target_milestone_list = set(available_target_milestone_list) - set(target_milestone_list)
            if len(invalid_target_milestone_list) > 0:
                logger.warning(f"invalid target milestone found: {invalid_target_milestone_list}, they will be ignored")
                # remove invalid target milestone from target_milestone_list
                for i in invalid_target_milestone_list:
                    target_milestone_list.remove(i)

        # 2. DFS to find all lineage from start_milestone to target_milestone_list
        lineage_dict = {}
        for target_milestone in target_milestone_list:
            # find shortest path of start_milestone to target_milestone as lineage
            sp = nx.shortest_path(G, source=start_milestone, target=target_milestone, weight="length")
            spl_dict = {start_milestone: 0}
            for i in range(len(sp) - 1):
                spl_dict[sp[i + 1]] = spl_dict[sp[i]] + G[sp[i]][sp[i + 1]].get("length", 1)
            logger.debug(f"shortest path from '{start_milestone}' to '{target_milestone}': {sp}")

            # extract cells along the lineage
            df_list = []
            for i in range(len(sp) - 1):
                df = mw.progressions.query(f"`from` == '{sp[i]}' & `to` == '{sp[i+1]}'")
                df_list.append(df)
            lineage_progressions = pd.concat(df_list)
            lineage_cell_id_list = lineage_progressions["cell_id"].tolist()

            lineage_dict[target_milestone] = lineage_cell_id_list

        # simple case: binary tree structure, lineage: A->B->C, A->B->D
        # lineage_dict = {
        #     "Alpha": [0, 1, 2],
        #     "Beta": [0, 1, 3],
        # }

        return lineage_dict

    def update_uns_cafe(self):
        # update .uns["cafe"]
        self.uns["cafe"] = self.cafe_dict

    def write_h5ad(self, filename):
        """Write the FateAnnData object to an h5ad file.

        This method temporarily serializes complex objects (like `MilestoneWrapper` and
        `WaypointWrapper` in `trajectory_history_dict`) into dictionaries/strings so they
        can be stored in the AnnData `.uns` slot, writes the file, and then restores the
        original objects.

        Args:
            filename (str): The filename to write to.
        """

        # the h5ad file will not only be read by CellFateExplorer, but also by scanpy.
        def serialize_trajectory_dict(self, model_name=None, delete_raw_wrapper_dict=True):
            # serialize trajectory for h5ad save
            logger.debug(f"serialize trajectory dict: '{model_name}'")
            trajectory_dict = self.get_trajectory_dict(model_name).copy()
            # transfer milestone object to dict
            milestone_wrapper = trajectory_dict.get("milestone_wrapper", None)
            if milestone_wrapper is not None and isinstance(milestone_wrapper, MilestoneWrapper):
                if hasattr(milestone_wrapper, "_milestone_network_G"):
                    # networkX.Graph object cannot be serialized, need to be remove from attribute.
                    delattr(milestone_wrapper, "_milestone_network_G")
                trajectory_dict["milestone_wrapper"] = milestone_wrapper.__dict__  # TODO: 保存时__dict__会修改category为int, 待修复
            # transfer waypoint object to dict
            waypoint_wrapper = trajectory_dict.get("waypoint_wrapper", None)
            if waypoint_wrapper is not None:
                if hasattr(waypoint_wrapper, "milestone_wrapper"):
                    # MilestoneWrapper object need to be remove from attribute
                    delattr(waypoint_wrapper, "milestone_wrapper")
                waypoint_wrapper.waypoints = waypoint_wrapper.waypoints.replace(
                    {None: ""}
                )  # fill the None value with empty string in milestone_id column
                trajectory_dict["waypoint_wrapper"] = waypoint_wrapper.__dict__
            # raw_wrapper_dict is complex, skip it
            if "raw_wrapper_dict" in trajectory_dict:
                logger.debug(f"delete raw_wrapper_dict in serialized trajectory dict: '{model_name}'")
                trajectory_dict["raw_wrapper_dict"] = {}
            return trajectory_dict

        raw_all_trajectory_dict = self.trajectory_history_dict.copy()
        for k in self.get_all_model_name(parse=False):
            std = serialize_trajectory_dict(self, k)
            self.set_trajectory_dict(std, k)
        super().write(filename)
        logger.debug(f"write h5ad to '{filename}'")
        self.trajectory_history_dict = raw_all_trajectory_dict  # recover raw trajectory dict
        logger.debug("recovery all raw trajectory dict")

    def check_result_dir(self, dirname=None):
        # TODO: check result dir for method run result
        # log: all workflow log, .log.
        # trajectory_dict: milestone and waypoint wrapper object in self.cfe_dict, .pkl.
        # metric: metric result, csv file.
        # h5ad: original method backend result, .h5ad.
        # image: plot function result, .png(easy), .pdf(for Adobe Illustrator)
        if dirname is None:
            dirname = os.path.join(settings.result_dir, ".cafe", self.id)

        subdirs = [
            "log",  # (.log)    all workflow log.
            "trajectory_history",  # (.pkl)    trajectory_dict storage
            "metric",  # (.csv)    milestone and waypoint wrapper object in self.cfe_dict["trajectory_history"]
            "h5ad",  # (.h5ad)   original h5ad files
            "img",  # (.png/.pdf for Adobe Illustrator) image outputs
            "benchmark",  # benchmark result
        ]

        for subdir in subdirs:
            subdir_path = os.path.join(dirname, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
                logger.debug(f"Created directory: '{subdir_path}'")

        self.result_dir = dirname
        self.log_dir = os.path.join(dirname, "log")
        self.trajectory_history_dir = os.path.join(dirname, "trajectory_history")
        self.metric_dir = os.path.join(dirname, "metric")
        self.h5ad_dir = os.path.join(dirname, "h5ad")
        self.image_dir = os.path.join(dirname, "img")
        self.benchmark_dir = os.path.join(dirname, "benchmark")

        # for save h5ad conveniently
        self.uns["cafe"]["dir"] = {
            "result_dir": self.result_dir,
            "log_dir": self.log_dir,
            "trajectory_history_dir": self.trajectory_history_dir,
            "metric_dir": self.metric_dir,
            "h5ad_dir": self.h5ad_dir,
            "image_dir": self.image_dir,
            "benchmark_dir": self.benchmark_dir,
        }

    def write_trajectory_dict(self, dirname=None, model_name_list=None):
        """Save trajectory dictionaries to pickle files.

        This method persists the trajectory history for specified models (or all valid models)
        into pickle files within the `trajectory_history` subdirectory of the result directory.

        Args:
            dirname (str, optional): The directory to save results in. If None, uses `self.result_dir`.
            model_name_list (list, optional): List of model names to save. If None, saves all models
                returned by `get_all_model_name(parse=False)`.
        """
        # save all trajectory, one trajectory is a pkl file: .cafe/{self.id}/trajectory_history/{model_name}.pkl
        # TODO: move to check_result_dir
        if dirname is None:
            dirname = self.trajectory_history_dir
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if model_name_list is None:
            # default save all trajectory
            model_name_list = self.get_all_model_name(parse=False)
        else:
            # TODO: check if the trajectory is compatible with the fadata object
            pass

        for model_name in model_name_list:
            model_filename = f"{dirname}/{model_name}.pkl"
            logger.debug(f"write trajectory '{model_name}' to '{model_filename}'")
            trajectory_dict = self.get_trajectory_dict(model_name)  # check compatibility
            with open(model_filename, "wb") as f:
                pickle.dump(trajectory_dict, f)

    def load_trajectory_dict(self, model_name_list: list[str] | str = None, dirname: str = None, backend: str = None):
        """Load trajectory dictionaries from pickle files.

        Restores trajectory history data from previously saved pickle files.

        Args:
            model_name_list (list[str] | str, optional): List of model names (or a single name) to load.
                If None/empty, attempts to load all .pkl files in the trajectory directory.
            dirname (str, optional): The directory to load results from. If None, uses `self.result_dir`.
            backend (str, optional): Backend to use (e.g., 'pickle'). Currently only supports pickle structure.

        Raises:
            FileNotFoundError: If the user-specified dirname does not exist or contain a 'trajectory_history' folder.
        """
        if dirname is None:
            dirname = self.trajectory_history_dir
        if not os.path.exists(dirname):
            raise Exception(f"directory '{dirname}' not found!")

        if model_name_list is None:
            # default load all trajectory in the dir
            model_name_list = [i.replace(".pkl", "") for i in os.listdir(dirname)]
            if backend is not None:
                # filter by backend
                filtered_model_name_list = []
                for model_name in model_name_list:
                    if model_name == "ref":
                        continue
                    # model name format: method_name-backend
                    now_backend = model_name.split("__")[1].split("-")[1]
                    if now_backend == backend:
                        filtered_model_name_list.append(model_name)
                model_name_list = filtered_model_name_list
        elif isinstance(model_name_list, str):
            model_name_list = [model_name_list]
        else:
            # TODO: Check if the trajectory is compatible with the data
            pass

        for model_name in model_name_list:
            if self.get_trajectory_dict(model_name) is not None:
                logger.debug(f"trajectory '{model_name}' already exists in the fadata object, skip loading")
                continue
            model_filename = f"{dirname}/{model_name}.pkl"
            logger.debug(f"load trajectory '{model_name}' from '{model_filename}'")
            with open(model_filename, "rb") as f:
                trajectory_dict = pickle.load(f)
            self.set_trajectory_dict(trajectory_dict, model_name)

    def remove_trajectory_dict(self, model_name_list: list[str] | str):
        if isinstance(model_name_list, str):
            model_name_list = [model_name_list]
        for model_name in model_name_list:
            if model_name in self.trajectory_history_dict:
                del self.trajectory_history_dict[model_name]
                self.model_name = "ref"
                logger.debug(f"remove trajectory '{model_name}' from trajectory_history_dict")
            else:
                logger.warning(f"trajectory '{model_name}' not found in trajectory_history_dict, skip remove")

    def recovery_external_data(self, model_name=None):
        external_data = self.get_raw_wrapper_dict(model_name).get("external_data")
        if external_data is None:
            logger.warning("no external data found in raw_wrapper_dict, return self")
            return self
        else:
            from ..util.anndata_attribute import recovery_external_data

            new_adata = recovery_external_data(self, external_data)
            return new_adata

    def clear_log():
        # clear log in cafe_dict
        pass

    def launch_cellxgene(self, tmp_filename=None, trajectory=False, port=5005, conda_env="cafe"):  # if show trajectory
        """Launch cellxgene to visualize the FateAnnData object.

        This function saves the current object to a temporary h5ad file and launches cellxgene
        for interactive visualization. It supports a custom mode for trajectory visualization.

        Args:
            tmp_filename (str, optional): Path for the temporary h5ad file. Defaults to "current_dir/.tmp.h5ad".
            trajectory (bool, optional): Whether to launch in trajectory visualization mode (requires special dev environment). Defaults to False.
            port (int, optional): Port to run the cellxgene server on. Defaults to 5005.
        """
        import os
        import subprocess
        import threading
        import time
        import webbrowser

        def print_output(pipe, prefix):
            """print output from a pipe"""
            for line in iter(pipe.readline, ""):
                if line:
                    logger.debug(f"{prefix}{line.rstrip()}")
            pipe.close()

        # 1. save as tmp.h5ad
        if tmp_filename is None:
            tmp_filename = f"{os.getcwd()}/.tmp.h5ad"
        self.write_h5ad(tmp_filename)
        logger.debug(f"write h5ad to {tmp_filename}")
        logger.debug("-" * 50)

        # 2. launch cellxgene
        # TODO: detect if cellxgene-cafe plugin is available, if not, launch normal cellxgene
        cmd = f"conda run -n {conda_env} --no-capture-output cellxgene launch --port {port} {tmp_filename}"  # conda run
        # # construct command
        # if trajectory:
        #     # TODO: local frontend and backend development version need be packaged
        #     # TODO: cxgxf打包后要能够一键执行
        #     # client_cmd = "cd /home/huang/PyCode/scRNA/CellXGene/cellxgene/client && make start-frontend"
        #     # subprocess.Popen(client_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) # frontend: react, ignore output
        #     # server_cmd = "cd /home/huang/PyCode/scRNA/CellXGene/cellxgene/client && make start-server"
        #     # process = subprocess.Popen(server_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) # backend: flask
        #     # logger.info("cellxgene with trajectory must run on port: 3000")
        #     # port = 3000
        #     # conda_env = "cafe" # 在当前环境下
        #     # cmd = f"conda run -n {conda_env} --no-capture-output cellxgene launch {tmp_filename} --port {port}"  # conda run
        #     # cmd = f"DATASET={tmp_filename}"  # dataset
        #     # cmd += f" & CXG_SERVER_PORT={5005}"  # server port
        #     # cmd += f" & CXG_CLIENT_PORT={port}"  # client port, web interface port
        #     # cmd += " & cd /root/PyCode/scRNA/CellFateExplorer/cafe-cellxgene/cellxgene"
        #     # cmd += " & make start-dev"
        #     # cellxgene with trajectory need use local development version
        #     cmd = "cd /root/PyCode/scRNA/CellFateExplorer/cafe-cellxgene/cellxgene && "
        #     cmd += f"DATASET={tmp_filename} CXG_SERVER_PORT={5005} CXG_CLIENT_PORT={port} make start-dev"
        # else:
        #     conda_env = "cellxgene"
        #     cmd = f"conda run -n {conda_env} --no-capture-output cellxgene launch {tmp_filename} --port {port}"  # conda run
        #     # conda activate + conda_env (usually use but not valid here)
        #     # cmd =  f"conda activate {conda_env} && cellxgene launch {tmp_filename} --port {port}"
        # # execuate command (NOTE: python_function can be executed in this way by conda)
        logger.debug(f"execute command: {cmd}")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        threading.Thread(target=print_output, args=(process.stdout, "[stdout]"), daemon=True).start()
        threading.Thread(target=print_output, args=(process.stderr, "[stderr]"), daemon=True).start()
        # open browser (NOTE: refresh browser if not valid)
        host = "127.0.0.1"
        time.sleep(5)  # wait for server to start
        if process.poll() is None:
            url = f"http://{host}:{port}"
            logger.info(f"🌐 Server start at: {url}")
            webbrowser.open(url)
            logger.debug("📝 Show cellxgene log")
        # wait for process
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.debug("-" * 50)
            logger.info("🛑 Server top!!!")
            process.terminate()
            process.wait()

        # 3. delete tmp.h5ad
        logger.debug(f"remove {tmp_filename}")
        os.remove(tmp_filename)

    def print_trajectory_data(self):
        from ..util.print_dict import print_dict

        print_dict(self.uns["cafe"], name="cafe")

    def check_model_name():
        pass

    def check_cluster(self, cluster=None):
        if cluster is None:
            if "cluster" not in self.prior_information:
                raise ValueError("parameter cluster is not provided and 'cluster' not found in self.prior_information")
            else:
                # extract from prior_information
                cluster = self.prior_information.get("cluster")
        else:
            if cluster not in self.obs:
                # check if cluster exists in self.obs
                raise ValueError(f"parameter cluster '{cluster}' not found in self.obs")
        return cluster

    def check_basis(self, basis=None):
        if basis is None:
            if "basis" not in self.prior_information:
                raise ValueError("parameter basis is not provided and 'basis' not found in self.prior_information")
            else:
                # extract from prior_information
                basis = self.prior_information.get("basis")
        else:
            if basis not in self.obsm:
                # check if basis exists in self.obsm
                raise ValueError(f"parameter basis '{basis}' not found in self.obsm")
        return basis

    # TODO: future work
    def combine_pseudotime_and_embedding(
        self,
        pseudotime,
        basis,
        cluster,
    ):
        # TODO: combine linear pseudotime and specific embedding to get cluster-level trajectory graph
        # A(0.1)->B(0.5)->C(0.8)/D(0.9),
        # Combing embedding space, A is root, B is the branch point, C/D is the terminal state.
        pass

    def combine_pseudotime_and_undircted_graph(
        self,
        pseudotime,
        # monocle2 result
    ):
        # TODO: combine linear pseudotime and undirected graph (such as monocle2 result) to get trajectory graph
        pass

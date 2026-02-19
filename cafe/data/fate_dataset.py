# TODO: config file for dataset in path cafe/data/dataset/...yaml
import pandas as pd
import scanpy as sc
from scipy import sparse as sp

from .. import settings
from .._logging import logger
from ..preprocess import subsample
from .fate_anndata import FateAnnData
from .fate_milestone_wrapper import MilestoneWrapper
from .fate_waypoint_wrapper import WaypointWrapper
# data_dir = settings.data_dir # need delay binding for data dir

def read_h5ad(*args, **kwargs):
    """Read a FateAnnData object from an h5ad file.

    This function wraps `scanpy.read_h5ad` to read the data and then converts it
    into a `FateAnnData` object. It also handles the deserialization of
    `trajectory_history_dict` (reconstructing `MilestoneWrapper` and `WaypointWrapper`
    objects from dictionaries).

    Args:
        *args: Variable length argument list passed to `scanpy.read_h5ad`.
        **kwargs: Arbitrary keyword arguments passed to `scanpy.read_h5ad`.

    Returns:
        FateAnnData: The loaded FateAnnData object with restored trajectory information.
    """
    adata = sc.read_h5ad(*args, **kwargs)
    fadata = FateAnnData.from_anndata(adata)

    def unserialize_trajectory_dict(fadata, model_name=None, recovery_raw_wrapper_dict=False):
        logger.debug(f"unserialize trajectory dict: '{model_name}'")
        trajectory_dict = fadata.get_trajectory_dict(model_name).copy()
        # parse milestone_wrapper
        milestone_wrapper = trajectory_dict.get("milestone_wrapper", None)
        if isinstance(milestone_wrapper, dict):
            # use object.__new__ to avoid __init__ function
            logger.debug(f"parse 'MilestoneWrapper' object for {model_name}")
            milestone_wrapper_obj = object.__new__(MilestoneWrapper)
            for k, v in milestone_wrapper.items():
                milestone_wrapper_obj[k] = v
            trajectory_dict["milestone_wrapper"] = milestone_wrapper_obj
        # parse waypoint_wrapper
        waypoint_wrapper = trajectory_dict.get("waypoint_wrapper", None)
        if (waypoint_wrapper is not None) and isinstance(waypoint_wrapper, dict):
            logger.debug(f"parse 'WaypointWrapper' object for {model_name}")
            waypoint_wrapper_obj = object.__new__(WaypointWrapper)
            for k, v in waypoint_wrapper.items():
                waypoint_wrapper_obj[k] = v
            trajectory_dict["waypoint_wrapper"] = waypoint_wrapper_obj
        # raw_wrapper_dict is complex, skip it
        if recovery_raw_wrapper_dict and "raw_wrapper_dict" in trajectory_dict:
            logger.debug(f"skip recovery raw_wrapper_dict in serialized trajectory dict: '{model_name}'")
        return trajectory_dict

    for k in fadata.get_all_model_name(parse=False):
        utd = unserialize_trajectory_dict(fadata, k)
        fadata.set_trajectory_dict(utd, k)

    return fadata

def _create_fadata_from_file(
    filename: str,
    cluster: str,
    basis: str,
    id: str = None,
    prior_information: dict = {},
    subsample_kwargs: dict = {},  # subsample args
    milestone_network: pd.DataFrame = None,
) -> FateAnnData:
    """Create a FateAnnData object from a file with specific configuration.

    This helper function reads a file (supporting h5ad), applies subsampling,
    sets up prior information (like cluster and basis keys), and optionally
    adds a manual reference trajectory based on a milestone network.

    Args:
        filename (str): Path to the input file (e.g., .h5ad).
        cluster (str): The key in `.obs` representing cell clusters/types.
        basis (str): The key in `.obsm` representing the embedding (e.g., 'X_umap').
        id (str, optional): Unique identifier for the dataset. Defaults to None.
        prior_information (dict, optional): Dictionary of prior information to add to the object. Defaults to {}.
        subsample_kwargs (dict, optional): Arguments for subsampling (e.g., {'n_obs': 1000}). Defaults to {}.
        milestone_network (pd.DataFrame, optional): A DataFrame defining the topology of a reference trajectory.
            If provided, a manual trajectory will be added. Defaults to None.

    Returns:
        FateAnnData: The initialized FateAnnData object.
    """
    logger.debug(f"Reading data from '{filename}'...")
    adata = sc.read_h5ad(filename)
    adata = subsample(adata, **subsample_kwargs)
    adata.uns["id"] = id
    # use csc matrix to replace for accelerate dynverse docker running.
    if not sp.isspmatrix_csc(adata.X):
        logger.debug("transfer 'X' and 'Spliced' matrix from csr to csc for better dynverse docker performance")
        adata.X = adata.X.tocsc()
        adata.layers["spliced"] = adata.layers["spliced"].tocsc()

    fadata = FateAnnData.from_anndata(adata)

    # for dynverse docker running
    fadata.layers["expression"] = fadata.X
    fadata.layers["counts"] = fadata.X
    fadata.obs["raw_index"] = fadata.obs.index
    fadata.obs.index = [f"cell_{i:03d}" for i in range(fadata.shape[0])]
    fadata.uns["filename"] = filename  # for methods that need filename rather than 'AnnData' object, such as pyrovelocity, unitvelo

    logger.debug("add prior information...")
    fadata.add_prior_information(**prior_information)
    # start_cell = prior_information.get("start_cell", None)
    # if start_cell is not None:
    #     if start_cell in fadata.obs.index:
    #         logger.debug(f"add 'start_cell': '{start_cell}'", indent_level=2)
    #         fadata.add_prior_information(start_cell=start_cell)
    #     else:
    #         logger.warning(f"{start_cell} is not in '.obs.index', skip adding 'start_cell'", indent_level=2)
    if milestone_network is not None:
        logger.debug("add ref trajectory mannually...")
        fadata.add_trajectory_mannually(
            milestone_network=milestone_network,
            cluster=cluster,
            basis=basis,
        )
    return fadata


def read_dynverse_simulation_data(
    filename=None,
    **subsample_kwargs,
):
    # read dynverse simulation data and create FateAnnData object,
    if filename is None:
        filename = f"{settings.data_dir}/dynbenchmark/data/synthetic/dyntoy/bifurcating_1.rds"

    import rpy2.robjects as ro

    from ..util import rpy2_read  # rpy2 data structure transfer automatically

    rpy2_read

    r_script = f"""
        dataset <- readRDS("{filename}")
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
    fadata = FateAnnData(name=dataset["id"], X=X)
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
        milestone_network = dataset["milestone_network"].reset_index(drop=True)
        milestone_percentages = dataset["milestone_percentages"]
        divergence_regions = dataset["divergence_regions"]
        # progressions = dataset["progressions"]
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


def read_bifurcating_cellrank(
    filename="../../tests/data/bifurcating.h5ad",
    **subsample_kwargs,
):
    milestone_network = pd.DataFrame(
        data=[
            ["sA -> sB", "sB -> sBmid"],
            ["sB -> sBmid", "sBmid -> sC"],
            ["sB -> sBmid", "sBmid -> sD"],
            ["sBmid -> sC", "sC -> sEndC"],
            ["sBmid -> sD", "sD -> sEndD"],
        ],
        columns=["from", "to"],
    )
    prior_information = {
        # "start_milestone": "sA -> sB",
        "cluster": "lineage",
        "basis": "X_umap",
    }
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster=prior_information["cluster"],
        basis=prior_information["basis"],
        id="bifurcating_cellrank",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )
    return fadata


def read_bonemarrow(
    filename=None,
    **subsample_kwargs,  # subsample args
):
    """read case study dataset of palantir and scvelo: bone marrow"""
    if filename is None:
        filename = f"{settings.data_dir}/BoneMarrow/setty_bone_marrow.h5ad"

    milestone_network = pd.DataFrame(
        data=[
            ["HSC_1", "HSC_2"],
            ["HSC_2", "Precursors"],
            ["HSC_2", "CLP"],
            ["HSC_2", "Ery_1"],
            ["Precursors", "Mono_1"],
            ["Precursors", "DCs"],
            ["Mono_1", "Mono_2"],
            ["Ery_1", "Ery_2"],
            ["Ery_1", "Mega"],
        ],
        columns=["from", "to"],
    )
    prior_information = {
        "start_milestone": "HSC_1",
        "start_cell": "cell_4823",
        "cluster": "clusters",
        "basis": "X_tsne",
    }
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster=prior_information["cluster"],
        basis=prior_information["basis"],
        id="bonemarrow",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )
    return fadata


def read_erythroid_lineage(
    filename=None,
    **subsample_kwargs,
):
    if filename is None:
        filename = f"{settings.data_dir}/Gastrulation/erythroid_lineage.h5ad"

    milestone_network = pd.DataFrame(
        data=[
            ["Blood progenitors 1", "Blood progenitors 2"],
            ["Blood progenitors 2", "Erythroid1"],
            ["Erythroid1", "Erythroid2"],
            ["Erythroid2", "Erythroid3"],
        ],
        columns=["from", "to"],
    )
    prior_information = {
        "start_cell": "cell_903",
        "end_cell": "cell_6099",
        "cluster": "celltype",
        "basis": "X_umap",
    }
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster=prior_information["cluster"],
        basis=prior_information["basis"],
        id="erythroid_lineage",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )
    return fadata


# TODO: gastrulation_5000


def read_gastrulation_5000(
    filename=None,
    **subsample_kwargs,
):
    """read case study dataset: gastrulation_5000"""
    if filename is None:
        filename = f"{settings.data_dir}/Gastrulation/gastrulation_5000.h5ad"

    milestone_network = pd.DataFrame(
        data=[
            ["Epiblast", "Anterior Primitive Streak"],
            ["Anterior Primitive Streak", "Primitive Streak"],
            ["Blood progenitors 1", "Blood progenitors 2"],
            ["Blood progenitors 2", "Erythroid1"],
            ["Erythroid1", "Erythroid2"],
            ["Erythroid2", "Erythroid3"],
        ],
        columns=["from", "to"],
    )
    prior_information = {
        "cluster": "celltype",
        "basis": "X_umap",
    }
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster=prior_information["cluster"],
        basis=prior_information["basis"],
        id="gastrulation_5000",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )

    return fadata


def read_gastrulation(
    filename=None,
    **subsample_kwargs,
):
    """read case study dataset: gastrulation"""
    # 文献来源: https://www.nature.com/articles/s41586-019-1825-8
    # 轨迹参考：https://github.com/MarioniLab/EmbryoTimecourse2018/blob/master/analysis_scripts/atlas/8_graph_abstraction/graph_abstraction.ipynb
    # 其他资料：
    #   维基百科：https://zh.wikipedia.org/wiki/原肠胚形成
    #   YouTube视频：https://www.youtube.com/watch?v=w9tJ7UiLrQs
    # 外胚层（Ectoderm）：外层, 发育为表皮、神经嵴，以及之后会发育为神经系统的组织
    # 中胚层（Mesoderm）：中层，发育为真皮、脊髓、血管与血液、骨、肌肉，以及结缔组织
    # 内胚层（Endoderm）：内层，发育为消化系统和呼吸系统的上皮，比如肝和胰腺
    # 这里就能理解为何stavia要重新注释细胞了

    if filename is None:
        filename = f"{settings.data_dir}/Gastrulation/gastrulation.h5ad"

    # TODO:
    milestone_network = pd.DataFrame(
        data=[
            ["Epiblast", "Anterior Primitive Streak"],
            ["Anterior Primitive Streak", "Primitive Streak"],
            ["Blood progenitors 1", "Blood progenitors 2"],
            ["Blood progenitors 2", "Erythroid1"],
            ["Erythroid1", "Erythroid2"],
            ["Erythroid2", "Erythroid3"],
        ],
        columns=["from", "to"],
    )
    prior_information = {
        "cluster": "celltype",
        "basis": "X_umap",
    }

    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster=prior_information["cluster"],
        basis=prior_information["basis"],
        id="gastrulation",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )

    return fadata


def read_dentategyrus():
    # TODO:
    pass


def read_pancreas(filename=None, **subsample_kwargs):
    if filename is None:
        filename = f"{settings.data_dir}/Pancreas/endocrinogenesis_day15.h5ad"

    milestone_network = pd.DataFrame(
        data=[
            ["Ductal", "Ngn3 low EP"],
            ["Ngn3 low EP", "Ngn3 high EP"],
            ["Ngn3 high EP", "Pre-endocrine"],
            ["Pre-endocrine", "Alpha"],
            ["Pre-endocrine", "Beta"],
            ["Pre-endocrine", "Delta"],
            ["Pre-endocrine", "Epsilon"],
        ],
        columns=["from", "to"],
    )
    prior_information = {
        "start_cell": "cell_1103",
        "cluster": "clusters",
        "basis": "X_umap",
    }
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster=prior_information["cluster"],
        basis=prior_information["basis"],
        id="pancreas",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )

    return fadata


# correct name from pancrease to pancreas, remove pancrease in future version
read_pancrease = read_pancreas


def read_pancreas_cellrank(filename=None, **subsample_kwargs):
    if filename is None:
        filename = f"{settings.data_dir}/Pancreas/endocrinogenesis_day15.5_velocity_kernel.h5ad"
    """read cellrank case study dataset: pancrease"""

    milestone_network = pd.DataFrame(
        data=[
            ["Ngn3 low EP", "Ngn3 high EP"],
            ["Ngn3 high EP", "Fev+"],
            ["Fev+", "Alpha"],
            ["Fev+", "Beta"],
            ["Fev+", "Delta"],
            ["Fev+", "Epsilon"],
        ],
        columns=["from", "to"],
    )
    prior_information = {
        "start_cell": "cell_2366",
        "cluster": "clusters",
        "basis": "X_umap",
    }
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster=prior_information["cluster"],
        basis=prior_information["basis"],
        id="pancreas_cellrank",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )
    return fadata


read_pancrease_cellrank = read_pancreas_cellrank

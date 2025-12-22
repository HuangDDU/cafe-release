import pandas as pd
import scanpy as sc
from scipy import sparse as sp

from .. import settings
from .._logging import logger
from ..preprocess import subsample
from .fate_anndata import FateAnnData

# data_dir = settings.data_dir # need delay binding for data dir


def _create_fadata_from_file(
    filename: str,
    milestone_network: pd.DataFrame,
    cluster: str,
    basis: str,
    id: str = None,
    prior_information: dict = {},
    subsample_kwargs: dict = {},  # subsample args
) -> FateAnnData:
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
    start_cell = prior_information.get("start_cell", None)
    if start_cell is not None:
        if start_cell in fadata.obs.index:
            logger.debug(f"add 'start_cell': '{start_cell}'", indent_level=2)
            fadata.add_prior_information(start_cell=start_cell)
        else:
            logger.warning(f"{start_cell} is not in '.obs.index', skip adding 'start_cell'", indent_level=2)

    fadata.add_trajectory_mannually(
        milestone_network=milestone_network,
        cluster_key=cluster,
        basis=basis,
    )
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
    prior_information = {}
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster="lineage",
        basis="X_umap",
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
    prior_information = {}
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster="clusters",
        basis="X_tsne",
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
    }
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster="celltype",
        basis="X_umap",
        id="erythroid_lineage",
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
    prior_information = {"start_cell": "cell_1103"}
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster="clusters",
        basis="X_umap",
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
    prior_information = {"start_cell": "cell_2366"}
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster="clusters",
        basis="X_umap",
        id="pancreas_cellrank",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )
    return fadata


read_pancrease_cellrank = read_pancreas_cellrank

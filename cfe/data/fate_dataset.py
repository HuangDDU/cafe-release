import pandas as pd
import scanpy as sc
from scipy import sparse as sp

from .. import settings
from .._logging import logger
from ..preprocess import subsample
from .fate_anndata import FateAnnData

data_dir = settings.data_dir


def _create_fadata_from_file(
    filename: str,
    milestone_network: pd.DataFrame,
    cluster_key: str,
    basis: str,
    id: str = None,
    prior_information: dict = {},
    subsample_kwargs: dict = {},  # subsample args
) -> FateAnnData:
    logger.debug(f"Reading data from '{filename}'...")
    adata = sc.read_h5ad(filename)
    subsample(adata, **subsample_kwargs)
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

    start_cell = prior_information.get("start_cell", None)
    if start_cell is not None:
        if start_cell in fadata.obs.index:
            logger.debug(f"add prior information 'start_cell': '{start_cell}'")
            fadata.add_prior_information(start_cell=start_cell)
        else:
            logger.warning(f"{start_cell} is not in '.obs.index', skip adding 'start_cell'")

    fadata.add_trajectory_mannually(
        milestone_network=milestone_network,
        cluster_key=cluster_key,
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
        cluster_key="lineage",
        basis="X_umap",
        id="bifurcating_cellrank",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )
    return fadata


def read_bonemarrow(
    filename=f"{data_dir}/BoneMarrow/setty_bone_marrow.h5ad",
    cluster_key="clusters",
    basis="X_tsne",
    **subsample_kwargs,  # subsample args
):
    """read case study dataset of palantir and scvelo: bone marrow"""
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
        cluster_key="clusters",
        basis="X_tsne",
        id="bonemarrow",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )
    return fadata


def read_erythroid_lineage(
    filename=f"{data_dir}/Gastrulation/erythroid_lineage.h5ad",
    **subsample_kwargs,
):
    milestone_network = pd.DataFrame(
        data=[
            ["Blood progenitors 1", "Blood progenitors 2"],
            ["Blood progenitors 2", "Erythroid1"],
            ["Erythroid1", "Erythroid2"],
            ["Erythroid2", "Erythroid3"],
        ],
        columns=["from", "to"],
    )
    prior_information = {}
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster_key="celltype",
        basis="X_umap",
        id="erythroid_lineage",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )
    return fadata


def read_dentategyrus():
    # TODO:
    pass


def read_pancrease(filename=f"{data_dir}/Pancreas/endocrinogenesis_day15.h5ad", **subsample_kwargs):
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
        cluster_key="clusters",
        basis="X_umap",
        id="pancreas",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )

    return fadata


def read_pancrease_cellrank(filename=f"{data_dir}/Pancreas/endocrinogenesis_day15.5_velocity_kernel.h5ad", **subsample_kwargs):
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
    prior_information = {}
    fadata = _create_fadata_from_file(
        filename=filename,
        milestone_network=milestone_network,
        cluster_key="clusters",
        basis="X_umap",
        id="pancreas_cellrank",
        prior_information=prior_information,
        subsample_kwargs=subsample_kwargs,
    )
    return fadata

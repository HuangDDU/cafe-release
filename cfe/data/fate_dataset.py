import pandas as pd
import scanpy as sc
from scipy import sparse as sp

from .._logging import logger
from ..preprocess import subsample
from .fate_anndata import FateAnnData

# class FateDataset():

#     def __init__():
#         pass

#     @classmethod
#     def read_dynverse_simulation_data(
#         cls,
#         filename="synthetic/dyntoy/bifurcating_1.rds",
#         dir="/usr/share/CellFateExplorer/dynbenchmark/data/"
#     ):
#         # 读取Dynverse的模拟数据
#         pass

#     @classmethod
#     def read_anndata(cls, filename):
#         # 读取AnnData格式的数据
#         pass


# TODO: use decorator to subsample cell from dataset


def read_bifurcating_cellrank(
    filename="../../tests/data/bifurcating.h5ad",
    cluster_key="lineage",
    basis="X_umap",
    **kwargs,  # subsample args
):
    """simulation data from cellrank"""
    adata = sc.read_h5ad(filename)
    subsample(adata, **kwargs)

    fadata = FateAnnData.from_anndata(adata)

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
    fadata.add_trajectory_mannually(
        milestone_network=milestone_network,
        cluster_key=cluster_key,
        basis=basis,
    )
    return fadata


def read_bonemarrow(
    filename="/home/huang/PyCode/scRNA/data/BoneMarrow/setty_bone_marrow.h5ad",
    cluster_key="clusters",
    basis="X_tsne",
    **kwargs,  # subsample args
):
    """read case study dataset of palantir and scvelo: bone marrow"""

    adata = sc.read_h5ad(filename)
    subsample(adata, **kwargs)

    fadata = FateAnnData.from_anndata(adata)
    # fadata.obs.index = [f"cell_{i:03d}" for i in range(fadata.shape[0])]

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
    fadata.add_trajectory_mannually(
        milestone_network=milestone_network,
        cluster_key=cluster_key,
        basis=basis,
    )

    return fadata


def read_erythroid_lineage(
    filename="/home/huang/PyCode/scRNA/data/Gastrulation/erythroid_lineage.h5ad",
    n_obs=-1,
    cluster_key="celltype",
    basis="X_umap",
    **kwargs,  # subsample args
):
    adata = sc.read_h5ad(filename)
    subsample(adata, **kwargs)

    fadata = FateAnnData.from_anndata(adata)

    milestone_network = pd.DataFrame(
        data=[
            ["Blood progenitors 1", "Blood progenitors 2"],
            ["Blood progenitors 2", "Erythroid1"],
            ["Erythroid1", "Erythroid2"],
            ["Erythroid2", "Erythroid3"],
        ],
        columns=["from", "to"],
    )
    fadata.add_trajectory_mannually(
        milestone_network=milestone_network,
        cluster_key=cluster_key,
        basis=basis,
    )

    return fadata


def read_dentategyrus():
    # TODO:
    pass


def read_pancrease(
    filename="/home/huang/PyCode/scRNA/data/Pancreas/endocrinogenesis_day15.h5ad",
    basis="X_umap",
    cluster_key="clusters",
    **kwargs,  # subsample args
):
    """read scvelo case study dataset: pancrease"""

    adata = sc.read_h5ad(filename)
    adata = subsample(adata, **kwargs)

    # use csc matrix to replace for accelerate dynverse docker running.
    if not sp.isspmatrix_csc(adata.X):
        logger.debug("transfer 'X' and 'Spliced' matrix from csr to csc for better dynverse docker performance")
        adata.X = adata.X.tocsc()
        adata.layers["spliced"] = adata.layers["spliced"].tocsc()

    fadata = FateAnnData.from_anndata(adata)

    fadata.layers["expression"] = fadata.layers["spliced"]
    fadata.layers["counts"] = fadata.layers["spliced"]
    fadata.obs["raw_index"] = fadata.obs.index
    fadata.obs.index = [f"cell_{i:03d}" for i in range(fadata.shape[0])]
    fadata.uns["filename"] = filename  # for methods that need filename rather than 'AnnData' object, such as pyrovelocity, unitvelo

    # automatically extracted prior information: {'cluster': 'clusters', 'basis': 'X_umap'}
    # add prior information mannully,
    start_cell = "cell_1103"
    if start_cell in fadata.obs.index:
        logger.debug(f"add prior information 'start_cell': '{start_cell}'")
        fadata.add_prior_information(start_cell=start_cell)
    else:
        logger.warning(f"{start_cell} is not in '.obs.index', skip adding 'start_cell'")

    # add milestone mannually
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

    fadata.add_trajectory_mannually(
        milestone_network=milestone_network,
        cluster_key=cluster_key,
        basis=basis,
    )

    return fadata


def read_pancrease_cellrank(
    filename="/home/huang/PyCode/scRNA/data/Pancreas/endocrinogenesis_day15.5_velocity_kernel.h5ad",
    basis="X_umap",
    cluster_key="clusters",
    **kwargs,  # subsample args
):
    """read cellrank case study dataset: pancrease"""

    adata = sc.read_h5ad(filename)
    adata = subsample(adata, **kwargs)

    fadata = FateAnnData.from_anndata(adata)
    fadata.layers["expression"] = fadata.layers["spliced"]
    fadata.layers["counts"] = fadata.layers["spliced"]
    fadata.obs.index = [f"cell_{i:03d}" for i in range(fadata.shape[0])]

    # add milestone mannually
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
    fadata.add_trajectory_mannually(
        milestone_network=milestone_network,
        cluster_key=cluster_key,
        basis=basis,
    )

    return fadata

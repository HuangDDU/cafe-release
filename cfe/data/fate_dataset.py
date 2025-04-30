import pandas as pd
import scanpy as sc
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


# TODO: add to public directory
def read_bonemarrow(
    filename="/home/huang/PyCode/scRNA/data/BoneMarrow/setty_bone_marrow.h5ad",
    cluster_key="clusters",
    basis="X_tsne",
    n_obs=-1,
    save_cell_list=[],
):
    # read case study dataset of palantir and scvelo: bone marrow
    adata = sc.read_h5ad(filename)

    # TODO: abstractly extract from the function to common tools
    # subsample and save a subset of cells such as start or terminal state cells.
    if n_obs > 0:
        if len(save_cell_list) > 0:
            adata_save = adata[save_cell_list, :].copy()
            n_obs -= len(adata_save)
            adata = adata[~(adata.obs.index.map(lambda x: x in save_cell_list))].copy()
        else:
            adata_save = None

        sc.pp.subsample(adata, n_obs=n_obs)  # subsample

        if adata_save is not None:
            adata = sc.concat([adata_save, adata])

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
        columns=["from", "to"]
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
    basis="X_umap"
):
    # read case study dataset of palantir and scvelo: bone marrow
    adata = sc.read_h5ad(filename)
    if n_obs > 0:
        sc.pp.subsample(adata, n_obs=n_obs)  # subsample
    fadata = FateAnnData.from_anndata(adata)

    milestone_network = pd.DataFrame(
        data=[
            ["Blood progenitors 1", "Blood progenitors 2"],
            ["Blood progenitors 2", "Erythroid1"],
            ["Erythroid1", "Erythroid2"],
            ["Erythroid2", "Erythroid3"],
        ],
        columns=["from", "to"]
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
    n_obs=-1
):
    # read scvelo case study dataset: pancrease
    adata = sc.read_h5ad(filename)
    if n_obs > 0:
        sc.pp.subsample(adata, n_obs=n_obs)  # subsample
    fadata = FateAnnData.from_anndata(adata)
    fadata.layers["expression"] = fadata.layers["spliced"]
    fadata.layers["count"] = fadata.layers["spliced"]
    fadata.obs.index = [f"cell_{i:03d}" for i in range(fadata.shape[0])]

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
        columns=["from", "to"]
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
    n_obs=-1
):
    # read cellrank case study dataset: pancrease
    adata = sc.read_h5ad(filename)
    if n_obs > 0:
        sc.pp.subsample(adata, n_obs=n_obs)  # subsample
    fadata = FateAnnData.from_anndata(adata)
    fadata.layers["expression"] = fadata.layers["spliced"]
    fadata.layers["count"] = fadata.layers["spliced"]
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
        columns=["from", "to"]
    )
    fadata.add_trajectory_mannually(
        milestone_network=milestone_network,
        cluster_key=cluster_key,
        basis=basis,
    )

    return fadata

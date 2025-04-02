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


def read_palantir_bone_marrow(filename="/home/huang/PyCode/scRNA/data/BoneMarrow/setty_bone_marrow.h5ad", basis="X_tsne", n_obs=-1):
    # 读取palantir的示例骨髓数据
    # read palantir case study dataset: bone marrow
    adata = sc.read_h5ad(filename)
    if n_obs > 0:
        sc.pp.subsample(adata, n_obs=n_obs)  # subsample
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
    fadata.add_trajectory_mannually(milestone_network, basis=basis)

    return fadata


def read_scvelo_pancrease(filename="/home/huang/PyCode/scRNA/data/Pancreas/endocrinogenesis_day15.h5ad", basis="X_umap", n_obs=-1):
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
    fadata.add_trajectory_mannually(milestone_network)

    return fadata

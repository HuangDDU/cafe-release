import pandas as pd
import networkx as nx
import anndata as ad
import scanpy as sc


def cf_cell_mst(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # 1. 数据构造
    adata = adata.copy()
    cell_ids = adata.obs.index

    # 2. 执行PCA
    sc.pp.pca(adata, n_comps=parameters["ndim"])

    # 3. 直接构建细胞间的最小生成树
    sc.pp.neighbors(adata)
    G = nx.from_scipy_sparse_array(adata.obsp["distances"])  # 从稀疏矩阵构造图
    cell_mst = nx.minimum_spanning_tree(G, weight="weight")

    # 4. 提取并封装结果
    cell_graph = nx.to_pandas_edgelist(cell_mst, source="from", target="to").rename(columns={"weight": "length"})
    cell_graph["from"] = cell_graph["from"].apply(lambda x: cell_ids[x])
    cell_graph["to"] = cell_graph["to"].apply(lambda x: cell_ids[x])
    to_keep = pd.Series(data=True, index=cell_ids)

    trajectory_dict = {
        "cell_graph": cell_graph,
        "to_keep": to_keep,
    }
    return trajectory_dict

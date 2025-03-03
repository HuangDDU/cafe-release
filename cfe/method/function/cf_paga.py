#!/usr/local/bin/python3
import pickle

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvelo as scv


def cf_paga(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):

    # 1. 数据构造
    adata = adata.copy()
    # 提取先验知识和参数
    start_id = prior_information["start_id"]
    connectivity_cutoff = parameters.get("connectivity_cutoff", 0.5)
    cluster_key = "cf_paga_clusters"
    adata.obs[cluster_key] = prior_information["groups_id"]

    # 2. 预处理
    scv.pp.filter_and_normalize(adata)
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.diffmap(adata)

    # 3. 方法调用
    # PAGA调用
    sc.tl.paga(adata, groups=cluster_key)
    # 设置起点执行dpt
    adata.uns["iroot"] = np.where(adata.obs.index == start_id)[0][0]
    sc.tl.dpt(adata, n_dcs=2)

    # 4. 结果提取
    # (1)
    epsilon = 1e-3  # 后续对于非常小的数字缩放值
    # cell_ids = adata.obs.index.to_list()
    branch_ids = adata.obs[cluster_key].unique().to_list()
    # (2) branches
    branches = pd.DataFrame({
        "branch_id": branch_ids,
        "directed": True,
    })
    branches["length"] = adata.obs[[cluster_key, "dpt_pseudotime"]].groupby(cluster_key).apply(lambda x: x["dpt_pseudotime"].max() - x["dpt_pseudotime"].min() + epsilon).reset_index()[0]
    # (3) branch_network
    branch_network = pd.DataFrame(
        np.triu(adata.uns["paga"]["connectivities"].todense(), k=0),  # 保留上三角矩阵
        index=adata.obs[cluster_key].cat.categories,
        columns=adata.obs[cluster_key].cat.categories
    ).stack().reset_index()
    branch_network.columns = ["from", "to", "length"]
    branch_network = branch_network[branch_network["length"] >= connectivity_cutoff]  # 设置阈值过滤不显著的边
    average_pseudotime_dict = adata.obs.groupby(cluster_key)["dpt_pseudotime"].mean()

    def modify_milestone_network_direction(x):
        if average_pseudotime_dict[x["from"]] <= average_pseudotime_dict[x["to"]]:
            return x
        else:
            x["from"], x["to"] = x["to"], x["from"]
            return x
    branch_network.apply(modify_milestone_network_direction, axis=1)  # 调整边的方向
    # 按照from、to的伪时间顺序排列，方便后续milestone编号
    branch_network["from_pseudotime"] = branch_network["from"].apply(lambda x: average_pseudotime_dict[x])
    branch_network["to_pseudotime"] = branch_network["to"].apply(lambda x: average_pseudotime_dict[x])
    branch_network = branch_network.sort_values(["from_pseudotime", "to_pseudotime"])
    branch_network = branch_network[["from", "to"]].reset_index(drop=True)  # 只保留from, to列
    # (4) branch_progressions
    branch_progressions = pd.DataFrame({
        "cell_id": adata.obs.index,
        "branch_id": adata.obs[cluster_key],
        "percentage": adata.obs["dpt_pseudotime"]
    })
    # branch内部按照伪时间排序
    branch_progressions["percentage"] = branch_progressions.groupby("branch_id")["percentage"].apply(lambda x: (x - x.min()) / (x.max() - x.min() + epsilon)).values
    branch_progressions

    # # 5. 结果封装保存
    # fadata.add_trajectory_branch(
    #     branch_network=branch_network,
    #     branches=branches,
    #     branch_progressions=branch_progressions

    # )
    trajectory_dict = {
        "branch_network": branch_network,
        "branches": branches,
        "branch_progressions": branch_progressions,
        # "trajectory_type": "branch_network",  # TODO: 暂时直接写，后续要用方法对轨迹数据判断获得类型
    }
    return trajectory_dict


if __name__ == "__main__":
    # for docker 

    from parse_args import parse_args

    adata, prior_information, parameters, output_filename = parse_args()

    trajectory_dict = cf_paga(adata, prior_information, parameters)

    with open(output_filename, "wb") as f:
        pickle.dump(trajectory_dict, f)
    print("PAGA Finish!")

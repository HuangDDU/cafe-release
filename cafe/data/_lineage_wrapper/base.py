import pandas as pd

from ..._logging import logger
from ...util import project_to_segments


def build_trajectory_base(fadata, probability: pd.DataFrame, cluster_key: str = None, new_cluster_list: list = None):
    if cluster_key is None:
        # use new cluster (macrostate) list
        if new_cluster_list is None:
            raise ValueError("cluster_key and new_cluster_list cannot be None at the same time.")
        else:
            logger.debug("using new cluster/macrostate list")
            cluster_list = new_cluster_list
    else:
        # use cluster attribute in adata.obs
        cluster_list = fadata.obs[cluster_key].tolist()
    lineage_name_list = probability.columns.tolist()  # 终端状态聚类名称
    main_cluster_list = [i for i in list(set(cluster_list)) if i not in lineage_name_list]  # 主干道上的聚类名称

    # 概率空间的的里程碑计算
    cluster_probability = probability.copy()
    cluster_probability["cluster"] = cluster_list
    cluster_probability = cluster_probability.groupby("cluster").agg("mean")
    logger.debug(f"cluster_probability:\n{cluster_probability}", indent_level=3)

    # 寻找谱系串
    lineage_list_list = []
    for lineage_name in lineage_name_list:
        tmp_cluster_list = main_cluster_list + [lineage_name]
        lineage_list = cluster_probability.loc[tmp_cluster_list, lineage_name].sort_values().index.tolist()
        lineage_list_list.append(lineage_list)
    logger.debug(f"lineage_list_list:\n{lineage_list_list}", indent_level=3)

    # 合并谱系串计算得到milestone_network与divergence_regions
    # 最长公共前缀

    def get_prefix_cluster_list(lineage_list_list: list):
        prefix_cluster_list = []
        for cluster_list in zip(*lineage_list_list):
            if len(set(cluster_list)) == 1:
                prefix_cluster_list.append(cluster_list[0])
            else:
                break
        return prefix_cluster_list

    prefix_cluster_list = get_prefix_cluster_list(lineage_list_list)
    if len(prefix_cluster_list) == 0:
        logger.warning("cannot find common prefix cluster, use probability wrapper")
        fadata.add_trajectory_probability(end_state_probabilities=probability)
        return
    else:
        branch_cluster = prefix_cluster_list[-1]
        # milestone_network
        milestone_network = pd.DataFrame(
            columns=["from", "to"],
            data=list(zip(prefix_cluster_list[:-1], prefix_cluster_list[1:])) + [[branch_cluster, i] for i in lineage_name_list],
        )
        milestone_network["length"] = 1
        milestone_network["directed"] = True
        # divergence_regions
        divergence_id = "".join([branch_cluster] + lineage_name_list)
        divergence_regions = pd.DataFrame(
            {
                "milestone_id": [branch_cluster] + lineage_name_list,
                "divergence_id": divergence_id,
                "is_start": [True] + [False] * len(lineage_name_list),
            }
        )

        # 暂时直接投影计算
        # print("probability\n", probability)
        # print(cluster_probability.loc[milestone_network["from"],])
        # print(cluster_probability.loc[milestone_network["to"],])
        proj = project_to_segments(
            x=probability,
            segment_start=cluster_probability.loc[milestone_network["from"],],
            segment_end=cluster_probability.loc[milestone_network["to"],],
        )
        progressions = milestone_network.iloc[proj["segment"] - 1][["from", "to"]]
        progressions["cell_id"] = fadata.obs.index
        progressions["percentage"] = proj["progression"]
        progressions = progressions[["cell_id", "from", "to", "percentage"]].reset_index(drop=True)
        # print(progressions)

        return {
            "milestone_network": milestone_network,
            "divergence_regions": divergence_regions,
            "progressions": progressions,
        }

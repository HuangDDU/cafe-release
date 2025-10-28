# import numpy as np
import pandas as pd

from cfe.data import FateAnnData


def calculate_mapping(
    fadata: FateAnnData,
    grouping: str = "milestones",
    simplify: bool = False,
    ref_model: str = "ref",
    pred_model: str = "default",
) -> dict:
    """
    计算轨迹映射指标——只需一个 FateAnnData，通过 ref_model / pred_model 从
    fadata.uns['cfe']['trajectory_history_dict'] 中取出两条不同轨迹进行对比。

    Args:
        fadata: 包含多条轨迹的 FateAnnData
        grouping: 'milestones' 或 'branches'
        simplify: 是否先简化轨迹骨架
        ref_model: 参考轨迹在 trajectory_history_dict 中的 key
        pred_model: 预测轨迹在 trajectory_history_dict 中的 key

    Returns:
        {'recovery': ..., 'relevance': ..., 'F1': ...}
    """
    # 参数校验
    if grouping not in ["branches", "milestones"]:
        raise ValueError("grouping must be either 'branches' or 'milestones'")

    # 1. 取出所有历史轨迹字典
    hist = fadata.uns.get("cfe", {}).get("trajectory_history_dict", {})
    # 如果任一模型不存在，直接返回 0
    if ref_model not in hist or pred_model not in hist:
        return {"recovery": 0.0, "relevance": 0.0, "F1": 0.0}

    # 2. （可选）简化骨架
    if simplify:
        fadata.simplify_trajectory(ref_model)
        fadata.simplify_trajectory(pred_model)

    # 3. 分组用到的列名
    if grouping == "milestones":
        group_key = "_cfe_nm_group"
    elif grouping == "branches":
        group_key = "_cfe_te_group"
    else:
        raise ValueError("grouping must be either 'milestones' or 'branches'")

    # 为了不丢失原 model_name，先保存后还原
    orig_model = fadata.model_name

    # 4. 对“参考”轨迹做分组
    fadata.model_name = ref_model
    if grouping == "milestones":
        fadata.group_onto_nearest_milestones(cluster_key=group_key)
    else:
        fadata.group_onto_trajectory_edges(cluster_key=group_key)
    groups_ref = fadata.obs.groupby(group_key).apply(lambda df: set(df.index))

    # 5. 对“预测”轨迹做分组
    fadata.model_name = pred_model
    if grouping == "milestones":
        fadata.group_onto_nearest_milestones(cluster_key=group_key)
    else:
        fadata.group_onto_trajectory_edges(cluster_key=group_key)
    groups_pred = fadata.obs.groupby(group_key).apply(lambda df: set(df.index))

    # 恢复原来的 model_name
    fadata.model_name = orig_model

    # 6. 计算 Jaccard 矩阵
    jaccard = pd.DataFrame(index=groups_ref.index, columns=groups_pred.index, dtype=float)
    for rname, rcells in groups_ref.items():
        for pname, pcells in groups_pred.items():
            inter = len(rcells & pcells)
            uni = len(rcells | pcells)
            jaccard.loc[rname, pname] = (inter / uni) if uni > 0 else 0.0

    # 7. recovery, relevance, F1
    recovery = jaccard.max(axis=1).mean() if not jaccard.empty else 0.0
    relevance = jaccard.max(axis=0).mean() if not jaccard.empty else 0.0
    if (recovery + relevance) > 0:
        f1 = 2 * recovery * relevance / (recovery + relevance)
    else:
        f1 = 0.0

    return {"recovery": recovery, "relevance": relevance, "F1": f1}


def calculate_mapping_milestones(fadata: FateAnnData, return_type: str = "score", **kwargs) -> dict:
    """计算里程碑分组映射指标"""
    m = calculate_mapping(fadata, grouping="milestones", **kwargs)
    if return_type == "score":
        return m["F1"]
    else:
        return {f"{k}_milestones": v for k, v in m.items()}


def calculate_mapping_branches(fadata: FateAnnData, return_type: str = "score", **kwargs) -> dict:
    """计算分支分组映射指标"""
    m = calculate_mapping(fadata, grouping="branches", **kwargs)
    if return_type == "score":
        return m["F1"]
    else:
        return {f"{k}_branches": v for k, v in m.items()}

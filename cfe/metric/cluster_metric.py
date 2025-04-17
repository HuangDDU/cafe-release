import pandas as pd
import numpy as np
from cfe.data import FateAnnData

def calculate_mapping(
    fadata_ref: FateAnnData, 
    fadata_pred: FateAnnData,
    grouping: str = 'milestones',
    simplify: bool = False,
    ref_model: str = "default",
    pred_model: str = "default"
) -> dict:
    """
    计算轨迹映射指标

    Args:
        fadata_ref: 参考轨迹数据 (FateAnnData)
        fadata_pred: 预测轨迹数据 (FateAnnData)
        grouping: 分组方式，'milestones'（里程碑）或 'branches'（分支）
        simplify: 是否简化轨迹
        ref_model: 参考轨迹模型名称
        pred_model: 预测轨迹模型名称

    Returns:
        dict: 包含 recovery、relevance 和 F1 的映射指标
    """
    # 参数校验
    if grouping not in ['branches', 'milestones']:
        raise ValueError("grouping must be either 'branches' or 'milestones'")
    
    # 检查轨迹数据是否存在
    if not fadata_ref.uns.get("cfe", {}).get("trajectory_history_dict") or \
       not fadata_pred.uns.get("cfe", {}).get("trajectory_history_dict"):
        return {'recovery': 0, 'relevance': 0, 'F1': 0}

    # 简化轨迹（注意：简化方法返回的是 MilestoneWrapper，不影响 fadata.obs）
    if simplify:
        # 此处仅调用简化方法以便内部更新存储，避免直接替换 FateAnnData 对象
        fadata_ref.simplify_trajectory(ref_model)
        fadata_pred.simplify_trajectory(pred_model)

    # 根据分组方式设置不同的分组键和调用相应的分组方法
    if grouping == 'branches':
        group_key = "_cfe_te_group"
        fadata_ref.group_onto_trajectory_edges(cluster_key=group_key)
        fadata_pred.group_onto_trajectory_edges(cluster_key=group_key)
    else:  # grouping == 'milestones'
        group_key = "_cfe_nm_group"
        fadata_ref.group_onto_nearest_milestones(cluster_key=group_key)
        fadata_pred.group_onto_nearest_milestones(cluster_key=group_key)

    # 构建分组Series：以分组键为分组依据，每个组对应一组 cell id 的集合
    def get_group_series(fadata: FateAnnData, key: str) -> pd.Series:
        return fadata.obs.groupby(key).apply(lambda df: set(df.index))
    
    groups_ref = get_group_series(fadata_ref, group_key)
    groups_pred = get_group_series(fadata_pred, group_key)

    # 计算Jaccard相似度矩阵
    jaccard_matrix = pd.DataFrame(
        index=groups_ref.index,
        columns=groups_pred.index,
        dtype=float
    )
    for ref_group, ref_cells in groups_ref.items():
        for pred_group, pred_cells in groups_pred.items():
            intersection = len(ref_cells.intersection(pred_cells))
            union = len(ref_cells.union(pred_cells))
            jaccard_matrix.loc[ref_group, pred_group] = intersection / union if union > 0 else 0.0

    # 计算 recovery 与 relevance
    recovery = jaccard_matrix.max(axis=1).mean()
    relevance = jaccard_matrix.max(axis=0).mean()
    f1 = 2 * (recovery * relevance) / (recovery + relevance) if (recovery + relevance) > 0 else 0.0

    return {'recovery': recovery, 'relevance': relevance, 'F1': f1}

def calculate_mapping_milestones(
    fadata_ref: FateAnnData,
    fadata_pred: FateAnnData,
    **kwargs
) -> dict:
    """计算里程碑分组映射指标"""
    metrics = calculate_mapping(fadata_ref, fadata_pred, grouping='milestones', **kwargs)
    return {f"{k}_milestones": v for k, v in metrics.items()}

def calculate_mapping_branches(
    fadata_ref: FateAnnData,
    fadata_pred: FateAnnData,
    **kwargs
) -> dict:
    """计算分支分组映射指标"""
    metrics = calculate_mapping(fadata_ref, fadata_pred, grouping='branches', **kwargs)
    return {f"{k}_branches": v for k, v in metrics.items()}
 
import time
import sys
import numpy as np
from scipy.stats import spearmanr
from cfe.data import FateAnnData

def calc_correlation(
    fadata_ref: FateAnnData, 
    fadata_pred: FateAnnData
) -> dict:
    """
    计算 FateAnnData 数据集与预测模型之间的地理距离相关性。
    
    要求 fadata_ref 和 fadata_pred 均已通过 add_trajectory() 添加轨迹信息，
    并调用 add_waypoints() 得到 waypoint_wrapper。

    Args:
        fadata_ref: 参考轨迹数据 (FateAnnData)
        fadata_pred: 预测轨迹数据 (FateAnnData)

    Returns:
        dict: 包含 'correlation'、'time_waypoint_geodesic_ref'、'time_waypoint_geodesic_pred' 和 'time_correlation'
    """
    metrics = {}
    # 验证数据结构
    """
    此处出自pydynverse
    #TODO:这里的逻辑可能需要补齐（不过暂时不管）
    if not is_wrapper_with_waypoint_cells(dataset):
        raise ValueError("Dataset must contain waypoint cells")
    if prediction is not None and not is_wrapper_with_waypoint_cells(prediction):
        raise ValueError("Prediction model must contain waypoint cells")
    """
    if fadata_pred is None:
        return {'correlation': 0.0}
    
    # 确保预测中的细胞都在参考数据中
    ref_cell_ids = list(fadata_ref.obs.index)
    pred_cell_ids = list(fadata_pred.obs.index)
    if not all(cell in ref_cell_ids for cell in pred_cell_ids):
        missing = set(pred_cell_ids) - set(ref_cell_ids)
        raise ValueError(f"Prediction contains unknown cells: {missing}")
    
    # 统一细胞顺序：采用参考数据的排序
    sorted_cells = sorted(ref_cell_ids)
    fadata_ref.obs = fadata_ref.obs.loc[sorted_cells]
    fadata_pred.obs = fadata_pred.obs.loc[sorted_cells]
    
    # 检查是否已添加 waypoint_wrapper
    if not fadata_ref.is_wrapped_with_waypoints or not fadata_pred.is_wrapped_with_waypoints:
        raise ValueError("Both FateAnnData objects must have been wrapped with waypoints (call add_waypoints()).")
    
    # 获取 waypoint_wrapper（当前模型默认使用 "default"）
    wp_ref = fadata_ref.waypoint_wrapper
    wp_pred = fadata_pred.waypoint_wrapper

    # 计算地理距离矩阵（调用 waypoint_wrapper 内部方法 _calculate_geodesic_distances）
    start_time = time.time()
    ref_dist = wp_ref._calculate_geodesic_distances()  # DataFrame：索引为 waypoint_id，列为 cell_id
    metrics['time_waypoint_geodesic_ref'] = time.time() - start_time

    start_time = time.time()
    pred_dist = wp_pred._calculate_geodesic_distances()
    metrics['time_waypoint_geodesic_pred'] = time.time() - start_time

    # 将无限值替换为最大浮点数
    max_float = sys.float_info.max
    def replace_inf(df, max_float):
        return df.replace([np.inf, -np.inf], max_float).to_numpy(dtype=np.float64)
    
    ref_arr = replace_inf(ref_dist, max_float)
    pred_arr = replace_inf(pred_dist, max_float)
    
    # 检查矩阵尺寸是否一致
    if ref_arr.shape != pred_arr.shape:
        raise RuntimeError(f"Distance matrix shape mismatch: {ref_arr.shape} vs {pred_arr.shape}")
    
    # 计算 Spearman 相关系数
    start_time = time.time()
    if np.unique(ref_arr).size == 1 or np.unique(pred_arr).size == 1:
        corr = 0.0
    else:
        corr, _ = spearmanr(ref_arr.flatten(), pred_arr.flatten())
        corr = max(corr, 0.0)
    metrics['correlation'] = corr
    metrics['time_correlation'] = time.time() - start_time
    
    return metrics

import time
import sys
from typing import Dict

import numpy as np
from scipy.stats import spearmanr

from cfe.data import FateAnnData

def calculate_correlation(
    fadata: FateAnnData,
    ref_model: str = "ref",
    pred_model: str = "default"
) -> Dict[str, float]:
    """
    计算两条已添加 waypoint 的轨迹（ref_model vs pred_model）之间的地理距离 Spearman 相关性。
    两个模型和它们对应的 waypoint_wrapper 都存储在同一个 FateAnnData.uns['cfe']['trajectory_history_dict'] 中。

    Args:
        fadata: 已经对多条轨迹都调用过 add_trajectory() 和 add_waypoints() 的 FateAnnData。
        ref_model: 参考模型的 key（trajectory_history_dict 中的字典键）。
        pred_model: 预测模型的 key。

    Returns:
        metrics: {
            'correlation': float,
            'time_waypoint_geodesic_ref': float,
            'time_waypoint_geodesic_pred': float,
            'time_correlation': float
        }
        若任一模型不存在或未生成 waypoint_wrapper，则直接返回 {'correlation': 0.0}。
    """
    metrics: Dict[str, float] = {"correlation": 0.0}

    # 1. 取出所有历史轨迹
    hist = fadata.uns.get("cfe", {}).get("trajectory_history_dict", {})
    if ref_model not in hist or pred_model not in hist:
        return metrics

    wp_ref = hist[ref_model].get("waypoint_wrapper")
    wp_pred = hist[pred_model].get("waypoint_wrapper")
    if wp_ref is None or wp_pred is None:
        raise ValueError(
            f"Both models must have waypoint_wrapper; "
            f"did you call add_waypoints() for '{ref_model}' and '{pred_model}'?"
        )

    # 2. 计算参考模型的 geodesic 距离并计时
    t0 = time.time()
    ref_dist_df = wp_ref._calculate_geodesic_distances()
    metrics["time_waypoint_geodesic_ref"] = time.time() - t0

    # 3. 计算预测模型的 geodesic 距离并计时
    t1 = time.time()
    pred_dist_df = wp_pred._calculate_geodesic_distances()
    metrics["time_waypoint_geodesic_pred"] = time.time() - t1

    # 4. 对齐：取两张表共有的 waypoint 行、所有细胞列
    common_wps = sorted(set(ref_dist_df.index) & set(pred_dist_df.index))
    if not common_wps:
        # 没有公共 waypoints，直接返回 0
        return metrics

    cells = sorted(fadata.obs.index.tolist())
    try:
        ref_arr = ref_dist_df.loc[common_wps, cells].to_numpy(dtype=np.float64)
        pred_arr = pred_dist_df.loc[common_wps, cells].to_numpy(dtype=np.float64)
    except KeyError as e:
        raise RuntimeError(
            f"细胞 ID 对齐失败："
            f"{e}. 请确保 obs.index 与 waypoint_distances 的列标签一致。"
        )

    # 5. 替换无穷大/NaN 为最大浮点数
    maxf = sys.float_info.max
    ref_arr[np.isinf(ref_arr) | np.isnan(ref_arr)]   = maxf
    pred_arr[np.isinf(pred_arr) | np.isnan(pred_arr)] = maxf

    # 6. 维度一致性检查
    if ref_arr.shape != pred_arr.shape:
        raise RuntimeError(
            f"距离矩阵维度不匹配："
            f"ref {ref_arr.shape} vs pred {pred_arr.shape}"
        )

    # 7. 计算 Spearman 相关并计时
    t2 = time.time()
    # 如果全部元素都相同，则相关性设为 0
    if np.unique(ref_arr).size == 1 or np.unique(pred_arr).size == 1:
        corr = 0.0
    else:
        corr, _ = spearmanr(ref_arr.flatten(), pred_arr.flatten())
        corr = max(corr, 0.0)  # 不要负值
    metrics["correlation"]      = corr
    metrics["time_correlation"] = time.time() - t2

    return metrics

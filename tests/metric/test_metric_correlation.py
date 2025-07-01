import numpy as np
import pandas as pd
import pytest

from cfe.data import FateAnnData
from cfe.metric.metric_correlation import calc_correlation


def get_test_fadata():
    """
    构造包含轨迹信息和 waypoint_wrapper 的 FateAnnData 对象。
    """
    # 构造细胞信息
    cell_ids = ["a", "b", "c", "d", "e"]
    obs = pd.DataFrame(index=cell_ids)
    uns = {"cfe": {"trajectory_history_dict": {"default": {}}}}
    fadata = FateAnnData(X=np.empty((len(cell_ids), 1)), obs=obs, uns=uns)

    # 构造里程碑网络
    milestone_network = pd.DataFrame({"from": ["W", "X", "X"], "to": ["X", "Y", "Z"], "length": [2, 3, 4], "directed": [True, True, True]})
    # 构造分叉区域
    divergence_regions = pd.DataFrame({"divergence_id": ["XYZ", "XYZ", "XYZ"], "milestone_id": ["X", "Y", "Z"], "is_start": [True, False, False]})
    # 构造里程碑百分比
    milestone_percentages = pd.DataFrame(
        {
            "cell_id": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
            "milestone_id": ["W", "X", "W", "X", "X", "Z", "Z", "X", "X", "Y"],
            "percentage": [0.9, 0.1, 0.2, 0.8, 0.8, 0.2, 0.1, 0.2, 0.3, 0.7],
        }
    )
    # 添加轨迹信息（内部会生成 MilestoneWrapper 并存储在 uns["cfe"]）
    fadata.add_trajectory(
        milestone_network=milestone_network,
        divergence_regions=divergence_regions,
        milestone_percentages=milestone_percentages,
    )
    # 添加 waypoint 信息（调用 add_waypoints，生成 WaypointWrapper）
    fadata.add_waypoints()

    return fadata


def test_calc_correlation():
    ref = get_test_fadata()
    pre = get_test_fadata()
    result = calc_correlation(ref, pre)
    # 当参考数据与预测数据完全相同时，相关性应为 1.0
    assert np.isclose(result["correlation"], 1.0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

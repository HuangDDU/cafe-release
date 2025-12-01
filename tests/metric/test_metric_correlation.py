import numpy as np
import pandas as pd
import pytest

from cafe.data import FateAnnData
from cafe.metric.metric_correlation import calculate_correlation


@pytest.fixture
def sample_fadata():
    # 构造细胞信息
    cell_ids = ["a", "b", "c", "d", "e"]
    obs = pd.DataFrame(index=cell_ids)
    uns = {"cafe": {"trajectory_history_dict": {}}}
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

    # 添加参考模型轨迹及 Waypoint
    fadata.model_name = "ref"
    fadata.add_trajectory(
        milestone_network=milestone_network,
        divergence_regions=divergence_regions,
        milestone_percentages=milestone_percentages,
    )
    fadata.add_waypoints()

    # 添加预测模型轨迹及 Waypoint（与参考相同）
    fadata.model_name = "pred"
    fadata.add_trajectory(
        milestone_network=milestone_network,
        divergence_regions=divergence_regions,
        milestone_percentages=milestone_percentages,
    )
    fadata.add_waypoints()

    return fadata


def test_calculate_correlation_identical(sample_fadata):
    # 参考与预测相同时，相关性应为 1.0
    metrics = calculate_correlation(sample_fadata, ref_model="ref", pred_model="pred", return_type="all")
    assert np.isclose(metrics["correlation"], 1.0)
    # 时间指标应存在且为浮点数
    assert isinstance(metrics["time_waypoint_geodesic_ref"], float)
    assert isinstance(metrics["time_waypoint_geodesic_pred"], float)
    assert isinstance(metrics["time_correlation"], float)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

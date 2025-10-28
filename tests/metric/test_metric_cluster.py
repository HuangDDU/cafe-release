import numpy as np
import pandas as pd
import pytest

from cfe.data import FateAnnData
from cfe.data.fate_milestone_wrapper import MilestoneWrapper
from cfe.metric.metric_cluster import calculate_mapping_milestones


@pytest.fixture
def sample_fadata():
    # 1. 创建一个 FateAnnData，有 6 个细胞
    cell_ids = ["a", "b", "c", "d", "e", "f"]
    obs = pd.DataFrame(index=cell_ids)
    uns = {"cfe": {"trajectory_history_dict": {}}}
    fadata = FateAnnData(X=np.empty((6, 1)), obs=obs, uns=uns)

    # 2. 构造“参考”里程碑 Wrapper，model_name='ref'
    ref_network = pd.DataFrame(
        {
            "from": ["W", "X", "X", "Z"],
            "to": ["X", "Y", "Z", "A"],
            "length": [1, 1, 1, 2],
            "directed": [True, True, True, True],
        }
    )
    ref_percent = pd.DataFrame(
        {
            "cell_id": ["a", "b", "b", "c", "c", "d", "e", "e", "e", "f", "f"],
            "milestone_id": ["W", "W", "X", "X", "Z", "Z", "X", "Y", "Z", "Z", "A"],
            "percentage": [1.0, 0.2, 0.8, 0.8, 0.2, 1.0, 0.3, 0.2, 0.5, 0.8, 0.2],
        }
    )
    mw_ref = MilestoneWrapper(milestone_network=ref_network, milestone_percentages=ref_percent)
    # 添加到 fadata
    fadata.model_name = "ref"
    fadata.milestone_wrapper = mw_ref

    # 3. 构造“预测”里程碑 Wrapper，model_name='pred'
    pred_network = ref_network.copy()
    pred_percent = pd.DataFrame(
        {
            "cell_id": ["a", "b", "b", "c", "c", "d", "f", "f", "e", "e"],
            "milestone_id": ["W", "W", "X", "X", "Z", "Z", "Z", "A", "X", "Y"],
            "percentage": [1.0, 0.2, 0.8, 0.8, 0.2, 1.0, 0.8, 0.2, 0.2, 0.8],
        }
    )
    mw_pred = MilestoneWrapper(milestone_network=pred_network, milestone_percentages=pred_percent)
    fadata.model_name = "pred"
    fadata.milestone_wrapper = mw_pred

    return fadata


def test_calculate_mapping_milestones(sample_fadata):
    # 计算里程碑映射（不简化）
    result = calculate_mapping_milestones(sample_fadata, simplify=False, ref_model="ref", pred_model="pred", return_type="all")
    # 预期值
    expected_recovery = 8 / 9
    expected_relevance = 3 / 4
    expected_f1 = 48 / 59

    assert result["recovery_milestones"] == pytest.approx(expected_recovery, rel=1e-2)
    assert result["relevance_milestones"] == pytest.approx(expected_relevance, rel=1e-2)
    assert result["F1_milestones"] == pytest.approx(expected_f1, rel=1e-2)

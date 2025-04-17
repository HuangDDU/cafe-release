import pytest
import pandas as pd
import numpy as np
from cfe.data import FateAnnData
from cfe.data.fate_milestone_wrapper import MilestoneWrapper
from cfe.metric.cluster_metric import calculate_mapping_milestones

# 构造参考数据 FateAnnData 对象
@pytest.fixture
def sample_fadata_ref():
    cell_ids = ['a', 'b', 'c', 'd', 'e', 'f']
    # 构造obs DataFrame，其索引用于存放cell_ids
    obs = pd.DataFrame(index=cell_ids)
    # 构造uns中存储cfe轨迹信息，至少保证trajectory_history_dict不为空
    uns = {"cfe": {"trajectory_history_dict": {"default": {}}}}
    # 构造 FateAnnData 对象（X数据可为空，这里仅用于测试）
    fadata = FateAnnData(X=np.empty((len(cell_ids), 1)), obs=obs, uns=uns)

    # 构造里程碑网络（参考数据）
    milestone_network = pd.DataFrame({
        "from": ["W", "X", "X", "Z"],
        "to":   ["X", "Y", "Z", "A"],
        "length": [1, 1, 1, 2],
        "directed": [True, True, True, True]
    })
    # 构造里程碑百分比（参考数据）
    milestone_percentages = pd.DataFrame({
        "cell_id": ["a", "b", "b", "c", "c", "d", "e", "e", "e", "f", "f"],
        "milestone_id": ["W", "W", "X", "X", "Z", "Z", "X", "Y", "Z", "Z", "A"],
        "percentage": [1.0, 0.2, 0.8, 0.8, 0.2, 1.0, 0.3, 0.2, 0.5, 0.8, 0.2]
    })
    # 利用 milestone_network 和 milestone_percentages 构造 MilestoneWrapper
    mw = MilestoneWrapper(milestone_network=milestone_network, milestone_percentages=milestone_percentages)
    # 将 MilestoneWrapper 保存到 FateAnnData 对象中（这里使用默认模型"default"）
    fadata.milestone_wrapper = mw
    return fadata

# 构造预测数据 FateAnnData 对象
@pytest.fixture
def sample_fadata_pred():
    cell_ids = ['a', 'b', 'c', 'd', 'e', 'f']
    obs = pd.DataFrame(index=cell_ids)
    uns = {"cfe": {"trajectory_history_dict": {"default": {}}}}
    fadata = FateAnnData(X=np.empty((len(cell_ids), 1)), obs=obs, uns=uns)

    milestone_network = pd.DataFrame({
        "from": ["W", "X", "X", "Z"],
        "to":   ["X", "Y", "Z", "A"],
        "length": [1, 1, 1, 2],
        "directed": [True, True, True, True]
    })
    # 构造预测数据的里程碑百分比（注意行的顺序和分布与参考数据略有不同）
    milestone_percentages = pd.DataFrame({
        "cell_id": ["a", "b", "b", "c", "c", "d", "f", "f", "e", "e"],
        "milestone_id": ["W", "W", "X", "X", "Z", "Z", "Z", "A", "X", "Y"],
        "percentage": [1.0, 0.2, 0.8, 0.8, 0.2, 1.0, 0.8, 0.2, 0.2, 0.8]
    })
    mw = MilestoneWrapper(milestone_network=milestone_network, milestone_percentages=milestone_percentages)
    fadata.milestone_wrapper = mw
    return fadata

def test_calculate_mapping_milestones(sample_fadata_ref, sample_fadata_pred):
    # 调用计算里程碑映射指标的函数，不进行轨迹简化（simplify=False）
    result = calculate_mapping_milestones(
        sample_fadata_ref,
        sample_fadata_pred,
        simplify=False,
        ref_model="default",
        pred_model="default"
    )
    # 预期值
    expected_recovery = 8 / 9
    expected_relevance = 3 / 4
    expected_F1 = 48 / 59

    # 使用 pytest.approx 对浮点数进行近似比较
    assert result["recovery_milestones"] == pytest.approx(expected_recovery, rel=1e-2)
    assert result["relevance_milestones"] == pytest.approx(expected_relevance, rel=1e-2)
    assert result["F1_milestones"] == pytest.approx(expected_F1, rel=1e-2)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
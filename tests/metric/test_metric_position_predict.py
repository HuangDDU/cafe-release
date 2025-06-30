import numpy as np
import pandas as pd
import pytest

from cfe.data import FateAnnData
from cfe.data.fate_milestone_wrapper import MilestoneWrapper
from cfe.metric.metric_position_predict import calculate_position_predict_fadata
from cfe.util.expand_matrix import expand_matrix


# 构造参考数据 FateAnnData 对象（测试中参考数据不变）
@pytest.fixture
def fadata_ref():
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
    milestone_percentages = pd.DataFrame({
        "cell_id": ["a", "b", "b", "c", "c", "d", "e", "e", "e", "f", "f"],
        "milestone_id": ["W", "W", "X", "X", "Z", "Z", "X", "Y", "Z", "Z", "A"],
        "percentage": [1.0, 0.2, 0.8, 0.8, 0.2, 1.0, 0.3, 0.2, 0.5, 0.8, 0.2]
    })
    mw = MilestoneWrapper(milestone_network=milestone_network,
                          milestone_percentages=milestone_percentages)
    fadata.milestone_wrapper = mw
    return fadata

# 构造预测数据 FateAnnData 对象，预测数据与参考数据完全一致
@pytest.fixture
def fadata_pred_same():
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
    # 与参考数据完全一致
    milestone_percentages = pd.DataFrame({
        "cell_id": ["a", "b", "b", "c", "c", "d", "e", "e", "e", "f", "f"],
        "milestone_id": ["W", "W", "X", "X", "Z", "Z", "X", "Y", "Z", "Z", "A"],
        "percentage": [1.0, 0.2, 0.8, 0.8, 0.2, 1.0, 0.3, 0.2, 0.5, 0.8, 0.2]
    })
    mw = MilestoneWrapper(milestone_network=milestone_network,
                          milestone_percentages=milestone_percentages)
    fadata.milestone_wrapper = mw
    return fadata

# 构造预测数据 FateAnnData 对象，预测数据存在偏差（例如整体加上 0.05），以观察指标变化
@pytest.fixture
def fadata_pred_diff():
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
    # 复制参考数据，然后对 percentage 加上 0.05（注意数据范围合理）
    milestone_percentages = pd.DataFrame({
        "cell_id": ["a", "b", "b", "c", "c", "d", "e", "e", "e", "f", "f"],
        "milestone_id": ["W", "W", "X", "X", "Z", "Z", "X", "Y", "Z", "Z", "A"],
        "percentage": [1.05, 0.25, 0.85, 0.85, 0.25, 1.05, 0.35, 0.25, 0.55, 0.85, 0.25]
    })
    mw = MilestoneWrapper(milestone_network=milestone_network,
                          milestone_percentages=milestone_percentages)
    fadata.milestone_wrapper = mw
    return fadata

def test_calculate_position_predict_same(fadata_ref, fadata_pred_same):
    """
    当预测数据与参考数据完全一致时，
    线性模型应完美拟合得到 lm_mse=0, lm_rsq=1, lm_nmse=1；
    随机森林回归由于内部近似，预期 rf_mse 约为 0.00656，rf_rsq 约为 0.89558，
    rf_nmse = 1 - (rf_mse / baseline_mse)。
    """
    result = calculate_position_predict_fadata(
        fadata_ref,
        prediction=fadata_pred_same,
        metrics=["rf_mse", "rf_rsq", "rf_nmse", "lm_mse", "lm_rsq", "lm_nmse"],
        model_name="default"
    )
    summary = result["summary"]

    # 使用 fadata_ref 内计算 baseline_mse 重现预期值
    cell_ids = fadata_ref.obs.index.tolist()
    gold_mp = fadata_ref.milestone_wrapper.milestone_percentages
    gold_milenet_m = pd.pivot_table(gold_mp, index="cell_id", columns="milestone_id",
                                    values="percentage", fill_value=0)
    gold_milenet_m = expand_matrix(gold_milenet_m, rownames=cell_ids)
    baseline_mse = np.mean([np.mean((gold_milenet_m[col] - gold_milenet_m[col].mean())**2)
                              for col in gold_milenet_m.columns])

    # 预期值（通过多次运行获得）：
    expected_rf_mse = 0.00656
    expected_rf_rsq = 0.89558
    expected_rf_nmse = 1 - (expected_rf_mse / baseline_mse)

    # 断言时给出适当容差
    assert summary["rf_mse"] == pytest.approx(expected_rf_mse, abs=1e-3)
    assert summary["rf_rsq"] == pytest.approx(expected_rf_rsq, abs=1e-3)
    assert summary["rf_nmse"] == pytest.approx(expected_rf_nmse, abs=1e-2)
    assert summary["lm_mse"] == pytest.approx(0, abs=1e-3)
    assert summary["lm_rsq"] == pytest.approx(1, abs=1e-3)
    assert summary["lm_nmse"] == pytest.approx(1, abs=1e-3)

def test_calculate_position_predict_diff(fadata_ref, fadata_pred_diff):
    """
    当预测数据与参考数据存在偏差时，指标将不再完美：
    – lm_mse 会大于 0，lm_rsq 会下降；
    – 随机森林模型同样会出现一定偏差。
    此测试主要用来观察输出结果，允许较宽的容差范围。
    """
    result = calculate_position_predict_fadata(
        fadata_ref,
        prediction=fadata_pred_diff,
        metrics=["rf_mse", "rf_rsq", "rf_nmse", "lm_mse", "lm_rsq", "lm_nmse"],
        model_name="default"
    )
    summary = result["summary"]
    assert summary["lm_mse"] > 0
    assert summary["lm_rsq"] < 1

if __name__ == "__main__":
    pytest.main(["-v", __file__])

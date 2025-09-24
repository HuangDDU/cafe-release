import numpy as np
import pandas as pd
import pytest

from cfe.data import FateAnnData
from cfe.data.fate_milestone_wrapper import MilestoneWrapper
from cfe.metric.metric_position_predict_v2 import calculate_position_predict
from cfe.util.expand_matrix import expand_matrix

CELL_IDS = ["a", "b", "c", "d", "e", "f"]


def make_fadata(milestone_percentages_ref, milestone_percentages_pred):
    # 共用 FateAnnData
    obs = pd.DataFrame(index=CELL_IDS)
    uns = {"cfe": {"trajectory_history_dict": {}}}
    fadata = FateAnnData(X=np.empty((len(CELL_IDS), 1)), obs=obs, uns=uns)

    # 里程碑网络固定
    mn = pd.DataFrame(
        {
            "from": ["W", "X", "X", "Z"],
            "to": ["X", "Y", "Z", "A"],
            "length": [1, 1, 1, 2],
            "directed": [True, True, True, True],
        }
    )
    # 参考模型
    mw_ref = MilestoneWrapper(milestone_network=mn, milestone_percentages=milestone_percentages_ref)
    fadata.model_name = "ref"
    fadata.milestone_wrapper = mw_ref

    # 预测模型
    mw_pred = MilestoneWrapper(milestone_network=mn, milestone_percentages=milestone_percentages_pred)
    fadata.model_name = "pred"
    fadata.milestone_wrapper = mw_pred

    return fadata


@pytest.fixture
def fadata_same():
    # 与参考完全一致
    mp = pd.DataFrame(
        {
            "cell_id": ["a", "b", "b", "c", "c", "d", "e", "e", "e", "f", "f"],
            "milestone_id": ["W", "W", "X", "X", "Z", "Z", "X", "Y", "Z", "Z", "A"],
            "percentage": [1.0, 0.2, 0.8, 0.8, 0.2, 1.0, 0.3, 0.2, 0.5, 0.8, 0.2],
        }
    )
    return make_fadata(mp, mp.copy())


@pytest.fixture
def fadata_diff():
    # 预测数据整体+0.05
    mp_ref = pd.DataFrame(
        {
            "cell_id": ["a", "b", "b", "c", "c", "d", "e", "e", "e", "f", "f"],
            "milestone_id": ["W", "W", "X", "X", "Z", "Z", "X", "Y", "Z", "Z", "A"],
            "percentage": [1.0, 0.2, 0.8, 0.8, 0.2, 1.0, 0.3, 0.2, 0.5, 0.8, 0.2],
        }
    )
    mp_pred = mp_ref.copy()
    mp_pred["percentage"] = mp_pred["percentage"] + 0.05
    return make_fadata(mp_ref, mp_pred)


def test_calculate_position_predict_same(fadata_same):
    # 相同数据下，LM 完美拟合，RF 近似
    result = calculate_position_predict(
        fadata_same,
        ref_model="ref",
        pred_model="pred",
        metrics=["rf_mse", "rf_rsq", "rf_nmse", "lm_mse", "lm_rsq", "lm_nmse"],
    )
    summary = result["summary"]

    # 计算 baseline_mse
    cell_ids = CELL_IDS
    gold = pd.pivot_table(
        fadata_same.uns["cfe"]["trajectory_history_dict"]["ref"]["milestone_wrapper"].milestone_percentages,
        index="cell_id",
        columns="milestone_id",
        values="percentage",
        fill_value=0,
    )
    gold = expand_matrix(gold, rownames=cell_ids)
    baseline = np.mean([((gold[col] - gold[col].mean()) ** 2).mean() for col in gold.columns])

    # 预期
    exp_rf_mse = 0.00656
    exp_rf_rsq = 0.89558
    exp_rf_nmse = 1 - exp_rf_mse / baseline

    # 修正的断言：lm_mse 而非 lf_mse
    assert summary["lm_mse"] == pytest.approx(0.0, abs=1e-6)
    assert summary["lm_rsq"] == pytest.approx(1.0, abs=1e-6)
    assert summary["lm_nmse"] == pytest.approx(1.0, abs=1e-6)

    assert summary["rf_mse"] == pytest.approx(exp_rf_mse, abs=1e-3)
    assert summary["rf_rsq"] == pytest.approx(exp_rf_rsq, abs=1e-3)
    assert summary["rf_nmse"] == pytest.approx(exp_rf_nmse, abs=1e-2)


def test_calculate_position_predict_diff(fadata_diff):
    result = calculate_position_predict(
        fadata_diff,
        ref_model="ref",
        pred_model="pred",
        metrics=["rf_mse", "rf_rsq", "rf_nmse", "lm_mse", "lm_rsq", "lm_nmse"],
    )
    summary = result["summary"]
    # diff 情况下，拟合不完美
    assert summary["lm_mse"] > 0
    assert summary["lm_rsq"] < 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])

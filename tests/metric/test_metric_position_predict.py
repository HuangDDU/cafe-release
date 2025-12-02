import numpy as np
import pandas as pd
import pytest

from cafe.data import FateAnnData
from cafe.data.fate_milestone_wrapper import MilestoneWrapper
from cafe.metric.metric_position_predict import calculate_position_predict

# from cafe.util.expand_matrix import expand_matrix

CELL_IDS = ["a", "b", "c", "d", "e", "f"]


def make_fadata(mp_ref, mp_pred):
    obs = pd.DataFrame(index=CELL_IDS)
    fadata = FateAnnData(X=np.empty((len(CELL_IDS), 1)), obs=obs, uns={"cafe": {"trajectory_history_dict": {}}})

    # 固定网络
    mn = pd.DataFrame(
        {
            "from": ["W", "X", "X", "Z"],
            "to": ["X", "Y", "Z", "A"],
            "length": [1, 1, 1, 2],
            "directed": [True, True, True, True],
        }
    )
    # 参考模型
    mw = MilestoneWrapper(milestone_network=mn, milestone_percentages=mp_ref)
    fadata.model_name = "ref"
    fadata.milestone_wrapper = mw
    # 预测模型
    mw2 = MilestoneWrapper(milestone_network=mn, milestone_percentages=mp_pred)
    fadata.model_name = "pred"
    fadata.milestone_wrapper = mw2

    return fadata


@pytest.fixture
def fadata_same():
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
    mp_ref = pd.DataFrame(
        {
            "cell_id": ["a", "b", "b", "c", "c", "d", "e", "e", "e", "f", "f"],
            "milestone_id": ["W", "W", "X", "X", "Z", "Z", "X", "Y", "Z", "Z", "A"],
            "percentage": [1.0, 0.2, 0.8, 0.8, 0.2, 1.0, 0.3, 0.2, 0.5, 0.8, 0.2],
        }
    )
    mp_pred = mp_ref.copy()
    mp_pred["percentage"] += 0.05
    return make_fadata(mp_ref, mp_pred)


def test_same_data_RF_and_LM_valid_ranges(fadata_same):
    """相同数据下，RF 和 LM 在测试集上应给出合理的非负 MSE 和 [0,1] 内 R²、NMSE。"""
    out = calculate_position_predict(fadata_same, ref_model="ref", pred_model="pred")["summary"]

    # RF 指标
    assert out["rf_mse"] >= 0.0
    assert 0.0 <= out["rf_rsq"] <= 1.0
    assert 0.0 <= out["rf_nmse"] <= 1.0

    # LM 指标
    assert out["lm_mse"] >= 0.0
    assert 0.0 <= out["lm_rsq"] <= 1.0
    assert 0.0 <= out["lm_nmse"] <= 1.0


def test_diff_data_LM_not_perfect(fadata_diff):
    """当预测数据有偏差时，LM 在测试集上不应是完美拟合：MSE>0，R²<1。"""
    out = calculate_position_predict(fadata_diff, ref_model="ref", pred_model="pred")["summary"]
    assert out["lm_mse"] > 0
    assert out["lm_rsq"] < 1.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])

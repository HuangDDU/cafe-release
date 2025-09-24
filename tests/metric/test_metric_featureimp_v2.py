import numpy as np
import pandas as pd
import pytest

from cfe.data import FateAnnData
from cfe.data.fate_milestone_wrapper import MilestoneWrapper
from cfe.metric.metric_featureimp_v2 import (
    calculate_featureimp_cor,
    calculate_featureimp_enrichment,
    calculate_overall_feature_importance,
    fi_ranger_rf_tiny,
    get_expression,
    is_wrapper_with_trajectory,
)

# from cfe.util.expand_matrix import expand_matrix

# from scipy.sparse import issparse
# from scipy.stats import ks_2samp, ranksums
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error


np.random.seed(42)
CELL_IDS = [f"cell{i}" for i in range(1, 21)]
GENES = [f"Gene{i}" for i in range(1, 11)]


@pytest.fixture
def fadata_dataset():
    # 构建基础 FateAnnData 用于整体重要性测试
    m1 = np.random.uniform(0.6, 0.8, size=len(CELL_IDS))
    m2 = 1 - m1

    # 表达矩阵
    expr = np.zeros((len(CELL_IDS), len(GENES)))
    expr[:, 0] = m1 + np.random.normal(0, 0.005, len(CELL_IDS))
    expr[:, 1] = m2 + np.random.normal(0, 0.005, len(CELL_IDS))
    for j in range(2, len(GENES)):
        expr[:, j] = np.random.uniform(0.45, 0.75, len(CELL_IDS))
    expression = pd.DataFrame(expr, index=CELL_IDS, columns=GENES)

    obs = pd.DataFrame(index=CELL_IDS)
    uns = {"cfe": {"trajectory_history_dict": {}}}
    fa = FateAnnData(X=np.empty((len(CELL_IDS), 1)), obs=obs, uns=uns)
    fa.obsm = {"expression": expression}
    fa.prior_information = {"features_id": ["Gene1", "Gene2"]}

    # 里程碑网络与百分比
    mn = pd.DataFrame({"from": ["M1"], "to": ["M2"], "length": [1], "directed": [True]})
    mp = pd.DataFrame(
        {
            "cell_id": CELL_IDS * 2,
            "milestone_id": ["M1"] * len(CELL_IDS) + ["M2"] * len(CELL_IDS),
            "percentage": np.concatenate([m1, m2]),
        }
    )
    mw = MilestoneWrapper(milestone_network=mn, milestone_percentages=mp)
    fa.model_name = "ref"
    fa.milestone_wrapper = mw

    return fa


@pytest.fixture
def fadata_combined(fadata_dataset):
    # 在同一个 fadata 中添加 pred 模型，基于 ref 百分比加偏移
    fa = fadata_dataset
    m1 = fa.milestone_wrapper.milestone_percentages.query("milestone_id=='M1'")["percentage"].values
    pred_m1 = m1 + 0.3
    pred_m2 = 1 - pred_m1
    mn = fa.milestone_wrapper.milestone_network

    mp_pred = pd.DataFrame(
        {
            "cell_id": CELL_IDS * 2,
            "milestone_id": ["M1"] * len(CELL_IDS) + ["M2"] * len(CELL_IDS),
            "percentage": np.concatenate([pred_m1, pred_m2]),
        }
    )
    mw_pred = MilestoneWrapper(milestone_network=mn, milestone_percentages=mp_pred)
    fa.model_name = "pred"
    fa.milestone_wrapper = mw_pred

    return fa


@pytest.fixture
def fi_method_tiny():
    return fi_ranger_rf_tiny(num_trees=50, num_variables_per_split=2, num_samples_per_tree=5, min_node_size=2)


def test_get_expression_and_wrapper():
    obs = pd.DataFrame(index=["a", "b", "c"])
    uns = {"cfe": {"trajectory_history_dict": {}}}
    fa = FateAnnData(X=np.array([[1, 2], [3, 4], [5, 6]]), obs=obs, uns=uns)

    # obsm absent -> 使用 X
    expr_df = get_expression(fa)
    assert isinstance(expr_df, pd.DataFrame)
    assert list(expr_df.index) == ["a", "b", "c"]

    # wrapper 未设置
    assert not is_wrapper_with_trajectory(fa)

    # 手动添加 milestone_wrapper + 标志
    mw_dummy = MilestoneWrapper(
        milestone_network=pd.DataFrame({"from": ["A"], "to": ["A"], "length": [1], "directed": [False]}),
        milestone_percentages=pd.DataFrame({"cell_id": ["a", "b", "c"], "milestone_id": ["A"] * 3, "percentage": [1, 1, 1]}),
    )
    fa.milestone_wrapper = mw_dummy
    fa.is_wrapped_with_trajectory = True
    assert is_wrapper_with_trajectory(fa)


def test_calculate_overall_feature_importance(fadata_dataset, fi_method_tiny):
    overall = calculate_overall_feature_importance(fadata_dataset, expression_source="expression", fi_method=fi_method_tiny)
    assert "feature_id" in overall.columns
    assert "importance" in overall.columns
    top_feats = overall["feature_id"].iloc[:2].tolist()
    # 前两位应该是 Gene1/Gene2
    assert set(top_feats) <= set(["Gene1", "Gene2"])


def test_calculate_featureimp_cor_and_enrichment(fadata_combined, fi_method_tiny):
    cor = calculate_featureimp_cor(fadata_combined, ref_model="ref", pred_model="pred", expression_source="expression", fi_method=fi_method_tiny)
    assert 0 < cor["featureimp_cor"] <= 1
    assert 0 < cor["featureimp_wcor"] <= 1

    enr = calculate_featureimp_enrichment(
        fadata_combined, ref_model="ref", pred_model="pred", expression_source="expression", fi_method=fi_method_tiny
    )
    assert 0 < enr["featureimp_ks"] <= 1
    assert 0 < enr["featureimp_wilcox"] <= 1


def test_featureimp_cor_insufficient(fadata_dataset, fi_method_tiny):
    fa = fadata_dataset
    # 构建仅2细胞的 pred wrapper
    small_cells = ["cell1", "cell2"]
    mn = fa.milestone_wrapper.milestone_network
    mp = pd.DataFrame({"cell_id": small_cells * 2, "milestone_id": ["M1", "M1"] + ["M2", "M2"], "percentage": [0.7, 0.8, 0.3, 0.2]})
    mw_pred = MilestoneWrapper(milestone_network=mn, milestone_percentages=mp)
    fa.model_name = "pred"
    fa.milestone_wrapper = mw_pred

    cor = calculate_featureimp_cor(fa, ref_model="ref", pred_model="pred", fi_method=fi_method_tiny)
    val = cor["featureimp_cor"]
    val2 = cor["featureimp_wcor"]
    # 对于细胞不足的情况，相关性可能直接返回 0.0，也可能是 NaN
    assert (val == 0.0) or np.isnan(val)
    assert (val2 == 0.0) or np.isnan(val2)


def test_featureimp_enrichment_insufficient(fadata_dataset, fi_method_tiny):
    fa = fadata_dataset
    small_cells = ["cell1", "cell2"]
    mn = fa.milestone_wrapper.milestone_network
    mp = pd.DataFrame({"cell_id": small_cells * 2, "milestone_id": ["M1", "M1"] + ["M2", "M2"], "percentage": [0.7, 0.8, 0.3, 0.2]})
    mw_pred = MilestoneWrapper(milestone_network=mn, milestone_percentages=mp)
    fa.model_name = "pred"
    fa.milestone_wrapper = mw_pred

    enr = calculate_featureimp_enrichment(fa, ref_model="ref", pred_model="pred", fi_method=fi_method_tiny)
    assert enr["featureimp_ks"] == 0.0
    assert enr["featureimp_wilcox"] == 0.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])

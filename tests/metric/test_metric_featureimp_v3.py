import numpy as np
import pandas as pd
import pytest

from cfe.data import FateAnnData
from cfe.data.fate_milestone_wrapper import MilestoneWrapper
from cfe.metric.metric_featureimp_v3 import (
    calculate_feature_importances,
    calculate_featureimp_cor,
    calculate_featureimp_enrichment,
    calculate_milestone_feature_importance,
    calculate_overall_feature_importance,
    fi_ranger_rf_tiny,
    get_expression,
    is_wrapper_with_trajectory,
)

# from scipy.sparse import issparse
# from scipy.stats import ks_2samp, ranksums
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# from cfe.util.expand_matrix import expand_matrix

np.random.seed(42)
CELL_IDS = [f"cell{i}" for i in range(1, 21)]
GENES = [f"Gene{i}" for i in range(1, 11)]


@pytest.fixture
def fadata_dataset():
    m1 = np.random.uniform(0.6, 0.8, size=len(CELL_IDS))
    m2 = 1 - m1

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

    expr_df = get_expression(fa)
    assert isinstance(expr_df, pd.DataFrame)
    assert list(expr_df.index) == ["a", "b", "c"]

    assert not is_wrapper_with_trajectory(fa)

    mw_dummy = MilestoneWrapper(
        milestone_network=pd.DataFrame({"from": ["A"], "to": ["A"], "length": [1], "directed": [False]}),
        milestone_percentages=pd.DataFrame({"cell_id": ["a", "b", "c"], "milestone_id": ["A"] * 3, "percentage": [1, 1, 1]}),
    )
    fa.milestone_wrapper = mw_dummy
    fa.is_wrapped_with_trajectory = True
    assert is_wrapper_with_trajectory(fa)


@pytest.mark.skip(reason="TODO: fix")
def test_calculate_overall_feature_importance(fadata_dataset, fi_method_tiny):
    overall = calculate_overall_feature_importance(fadata_dataset, expression_source="expression", fi_method=fi_method_tiny)
    assert "feature_id" in overall.columns
    assert "importance" in overall.columns
    top_feats = overall["feature_id"].iloc[:2].tolist()
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
    small_cells = ["cell1", "cell2"]
    mn = fa.milestone_wrapper.milestone_network
    mp = pd.DataFrame({"cell_id": small_cells * 2, "milestone_id": ["M1", "M1"] + ["M2", "M2"], "percentage": [0.7, 0.8, 0.3, 0.2]})
    mw_pred = MilestoneWrapper(milestone_network=mn, milestone_percentages=mp)
    fa.model_name = "pred"
    fa.milestone_wrapper = mw_pred

    cor = calculate_featureimp_cor(fa, ref_model="ref", pred_model="pred", fi_method=fi_method_tiny)
    val = cor["featureimp_cor"]
    val2 = cor["featureimp_wcor"]
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


def test_calculate_feature_importances_simple():
    n = 100
    X = pd.DataFrame(np.random.randn(n, 3), columns=["f0", "f1", "f2"])
    noise = np.random.randn(n) * 0.01
    Y = pd.DataFrame({"y1": X["f0"] * 1.0 + noise, "y2": X["f1"] * 2.0 + noise})
    fi = calculate_feature_importances(X, Y, fi_method=fi_ranger_rf_tiny(num_trees=200))
    top_y1 = fi.query("predictor_id=='y1'").nlargest(1, "importance")["feature_id"].iat[0]
    assert top_y1 == "f0"
    top_y2 = fi.query("predictor_id=='y2'").nlargest(1, "importance")["feature_id"].iat[0]
    assert top_y2 == "f1"
    assert len(fi) == 6


def make_toy_fadata():
    # 三个基因，两个里程碑。20 个细胞
    cells = [f"c{i}" for i in range(20)]
    # m1 线性从 0.9 下降到 0.1，m2 = 1 - m1
    m1 = np.linspace(0.9, 0.1, 20)
    m2 = 1 - m1

    # G1 强相关 m1，G2 强相关 m2，G3 为随机噪声
    expr = pd.DataFrame(
        {"G1": m1 + np.random.normal(0, 0.01, 20), "G2": m2 + np.random.normal(0, 0.01, 20), "G3": np.random.rand(20)},
        index=cells,
    )

    mp = pd.DataFrame({"cell_id": cells * 2, "milestone_id": ["M1"] * 20 + ["M2"] * 20, "percentage": np.concatenate([m1, m2])})
    mn = pd.DataFrame({"from": ["M1"], "to": ["M2"], "length": [1], "directed": [True]})

    fa = FateAnnData(X=np.empty((20, 1)), obs=pd.DataFrame(index=cells), uns={"cfe": {"trajectory_history_dict": {}}})
    fa.obsm = {"expression": expr}
    mw = MilestoneWrapper(milestone_network=mn, milestone_percentages=mp)
    fa.model_name = "ref"
    fa.milestone_wrapper = mw
    return fa


def test_milestone_and_overall_importance():
    fa = make_toy_fadata()
    fi_method = fi_ranger_rf_tiny(num_trees=500)

    milimp = calculate_milestone_feature_importance(fa, fi_method=fi_method)
    assert set(milimp["milestone_id"].unique()) == {"M1", "M2"}
    assert set(milimp["feature_id"].unique()) == {"G1", "G2", "G3"}

    overall = calculate_overall_feature_importance(fa, fi_method=fi_method)
    assert list(overall.columns) == ["feature_id", "importance"]
    imp_list = overall["importance"].tolist()
    assert imp_list == sorted(imp_list, reverse=True)


def test_featureimp_cor_midrange():
    fa = make_toy_fadata()
    # 构造 pred wrapper：在 ref 的 mp 基础上加少量噪声
    ref_mp = fa.milestone_wrapper.milestone_percentages.copy()
    pred_mp = ref_mp.copy()
    pred_mp["percentage"] += np.random.normal(0, 0.05, size=len(pred_mp))
    mw2 = MilestoneWrapper(milestone_network=fa.milestone_wrapper.milestone_network, milestone_percentages=pred_mp)
    hist = fa.uns["cfe"]["trajectory_history_dict"]
    hist["ref"] = {"milestone_wrapper": fa.milestone_wrapper}
    hist["pred"] = {"milestone_wrapper": mw2}

    fa.prior_information = {"features_id": ["G1", "G2"]}

    cor = calculate_featureimp_cor(fa, ref_model="ref", pred_model="pred", fi_method=fi_ranger_rf_tiny(num_trees=500))
    assert 0.1 < cor["featureimp_cor"] < 0.9 or np.isnan(cor["featureimp_cor"])
    assert 0.1 < cor["featureimp_wcor"] < 0.9 or np.isnan(cor["featureimp_wcor"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])

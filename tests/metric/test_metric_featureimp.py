import numpy as np
import pandas as pd
import pytest

from cfe.data import FateAnnData
from cfe.data.fate_milestone_wrapper import MilestoneWrapper
from cfe.metric.metric_featureimp import (
    calculate_featureimp_cor,
    calculate_featureimp_enrichment,
    calculate_overall_feature_importance,
    fi_ranger_rf_tiny,
    get_expression,
    is_wrapper_with_trajectory,
)
from cfe.util.expand_matrix import expand_matrix

# 固定随机种子
np.random.seed(42)

# 生成基因名称，这里设定10个基因
def generate_gene_names(n=10):
    return [f"Gene{i}" for i in range(1, n+1)]

@pytest.fixture
def fadata_dataset():
    """
    构造一个与里程碑相关的 FateAnnData 对象（数据集），包含20个细胞和10个基因：
      - 里程碑设为2个：M1 和 M2；
      - 对每个细胞生成 M1 值在 [0.6, 0.8] 内，令 M2 = 1 - M1；
      - 表达矩阵中：
            Gene1 = M1 + 少量噪声，
            Gene2 = M2 + 少量噪声，
            其它基因（Gene3～Gene10）均匀随机生成于 [0.45, 0.75]（使噪声更明显一点）；
      - 先验重要特征设置为 Gene1 和 Gene2。
    """
    cell_ids = [f'cell{i}' for i in range(1, 21)]
    genes = generate_gene_names(10)

    # 生成里程碑百分比：M1取自[0.6,0.8]，M2 = 1 - M1
    m1 = np.random.uniform(0.6, 0.8, size=len(cell_ids))
    m2 = 1 - m1

    # 构造表达矩阵：
    expr = np.zeros((len(cell_ids), len(genes)))
    # Gene1与M1强相关，加上极小噪声
    expr[:, 0] = m1 + np.random.normal(0, 0.005, len(cell_ids))
    # Gene2与M2强相关，加上极小噪声
    expr[:, 1] = m2 + np.random.normal(0, 0.005, len(cell_ids))
    # Gene3～Gene10设置在[0.45,0.75]，变化较大一些，便于产生一定差异
    for j in range(2, len(genes)):
        expr[:, j] = np.random.uniform(0.45, 0.75, len(cell_ids))
    expression = pd.DataFrame(expr, index=cell_ids, columns=genes)

    obs = pd.DataFrame(index=cell_ids)
    uns = {"cfe": {"trajectory_history_dict": {"default": {}}}}
    fadata = FateAnnData(X=np.empty((len(cell_ids), 1)), obs=obs, uns=uns)
    fadata.obsm = {"expression": expression}
    fadata.prior_information = {"features_id": ["Gene1", "Gene2"]}

    # 里程碑网络
    milestone_network = pd.DataFrame({
        "from": ["M1"],
        "to": ["M2"],
        "length": [1],
        "directed": [True]
    })
    milestone_percentages = pd.DataFrame({
        "cell_id": cell_ids * 2,
        "milestone_id": ["M1"] * len(cell_ids) + ["M2"] * len(cell_ids),
        "percentage": np.concatenate([m1, m2])
    })
    mw = MilestoneWrapper(milestone_network=milestone_network, milestone_percentages=milestone_percentages)
    fadata.milestone_wrapper = mw
    return fadata

@pytest.fixture
def fadata_prediction():
    """
    构造预测的 FateAnnData 对象：
      - 基于 dataset 的数据，整体在表达矩阵与里程碑百分比上增加较大偏移（例如0.3），
        使得预测数据与真实数据既相关但又不完全一致，
        并且使得 Gene1 与 Gene2 具有明显更高的信号。
    """
    cell_ids = [f'cell{i}' for i in range(1, 21)]
    genes = generate_gene_names(10)

    # 生成基准 m1, m2
    m1 = np.random.uniform(0.6, 0.8, size=len(cell_ids))
    m2 = 1 - m1

    # 对于预测表达矩阵，将Gene1和Gene2加上较大偏移 0.3（而不是0.1），并加微量噪声
    expr = np.zeros((len(cell_ids), len(genes)))
    expr[:, 0] = m1 + 0.3 + np.random.normal(0, 0.005, len(cell_ids))
    expr[:, 1] = m2 + 0.3 + np.random.normal(0, 0.005, len(cell_ids))
    for j in range(2, len(genes)):
        expr[:, j] = np.random.uniform(0.45, 0.75, len(cell_ids))
    expression = pd.DataFrame(expr, index=cell_ids, columns=genes)

    obs = pd.DataFrame(index=cell_ids)
    uns = {"cfe": {"trajectory_history_dict": {"default": {}}}}
    fadata = FateAnnData(X=np.empty((len(cell_ids), 1)), obs=obs, uns=uns)
    fadata.obsm = {"expression": expression}
    fadata.prior_information = {"features_id": ["Gene1", "Gene2"]}

    # 对于预测里的里程碑百分比，增加较大偏移 0.3
    milestone_network = pd.DataFrame({
        "from": ["M1"],
        "to": ["M2"],
        "length": [1],
        "directed": [True]
    })
    pred_m1 = m1 + 0.3
    pred_m2 = 1 - pred_m1
    milestone_percentages = pd.DataFrame({
        "cell_id": cell_ids * 2,
        "milestone_id": ["M1"] * len(cell_ids) + ["M2"] * len(cell_ids),
        "percentage": np.concatenate([pred_m1, pred_m2])
    })
    mw = MilestoneWrapper(milestone_network=milestone_network, milestone_percentages=milestone_percentages)
    fadata.milestone_wrapper = mw
    return fadata



# 使用轻量级小型森林参数 (fi_ranger_rf_tiny)

@pytest.fixture
def fi_method_tiny():
    return fi_ranger_rf_tiny(num_trees=50, num_variables_per_split=2, num_samples_per_tree=5, min_node_size=2)


def test_calculate_overall_feature_importance(fadata_dataset, fi_method_tiny):
    overall = calculate_overall_feature_importance(fadata_dataset, expression_source="expression", fi_method=fi_method_tiny)
    assert "feature_id" in overall.columns
    assert "importance" in overall.columns
    assert overall["importance"].max() > 0
    assert overall["importance"].iloc[0] >= overall["importance"].iloc[-1]


def test_calculate_featureimp_cor(fadata_dataset, fadata_prediction, fi_method_tiny):
    cor_dict = calculate_featureimp_cor(
        dataset=fadata_dataset,
        prediction=fadata_prediction,
        expression_source="expression",
        fi_method=fi_method_tiny
    )
    assert "featureimp_cor" in cor_dict
    assert "featureimp_wcor" in cor_dict
    # 我们期望这两个值都大于0
    assert 0 < cor_dict["featureimp_cor"] <= 1
    assert 0 < cor_dict["featureimp_wcor"] <= 1

def test_calculate_featureimp_enrichment(fadata_dataset, fadata_prediction, fi_method_tiny):
    enrich_dict = calculate_featureimp_enrichment(
        dataset=fadata_dataset,
        prediction=fadata_prediction,
        expression_source="expression",
        fi_method=fi_method_tiny
    )
    assert "featureimp_ks" in enrich_dict
    assert "featureimp_wilcox" in enrich_dict
    # 期望 KS 和 Wilcoxon 返回值均在(0,1)之间（非边界）
    assert 0 < enrich_dict["featureimp_ks"] <= 1
    assert 0 < enrich_dict["featureimp_wilcox"] <= 1

# 边缘测试：当预测对象只有2个细胞时返回0
def test_featureimp_cor_insufficient_cells(fadata_dataset):
    cell_ids = ['cell1', 'cell2']
    obs = pd.DataFrame(index=cell_ids)
    uns = {"cfe": {"trajectory_history_dict": {"default": {}}}}
    fadata_pred = FateAnnData(X=np.empty((len(cell_ids), 1)), obs=obs, uns=uns)
    milestone_network = pd.DataFrame({"from": ["M1"], "to": ["M2"], "length": [1], "directed": [True]})
    milestone_percentages = pd.DataFrame({
        "cell_id": cell_ids * 2,
        "milestone_id": ["M1", "M1", "M2", "M2"],
        "percentage": [0.7, 0.8, 0.3, 0.2]
    })
    mw = MilestoneWrapper(milestone_network=milestone_network, milestone_percentages=milestone_percentages)
    fadata_pred.milestone_wrapper = mw

    cor_dict = calculate_featureimp_cor(
        dataset=fadata_dataset,
        prediction=fadata_pred,
        expression_source="expression"
    )
    assert cor_dict["featureimp_cor"] == 0
    assert cor_dict["featureimp_wcor"] == 0

def test_featureimp_enrichment_insufficient_cells(fadata_dataset):
    cell_ids = ['cell1', 'cell2']
    obs = pd.DataFrame(index=cell_ids)
    uns = {"cfe": {"trajectory_history_dict": {"default": {}}}}
    fadata_pred = FateAnnData(X=np.empty((len(cell_ids), 1)), obs=obs, uns=uns)
    milestone_network = pd.DataFrame({"from": ["M1"], "to": ["M2"], "length": [1], "directed": [True]})
    milestone_percentages = pd.DataFrame({
        "cell_id": cell_ids * 2,
        "milestone_id": ["M1", "M1", "M2", "M2"],
        "percentage": [0.7, 0.8, 0.3, 0.2]
    })
    mw = MilestoneWrapper(milestone_network=milestone_network, milestone_percentages=milestone_percentages)
    fadata_pred.milestone_wrapper = mw

    enrich_dict = calculate_featureimp_enrichment(
        dataset=fadata_dataset,
        prediction=fadata_pred,
        expression_source="expression"
    )
    assert enrich_dict["featureimp_ks"] == 0
    assert enrich_dict["featureimp_wilcox"] == 0

if __name__ == "__main__":
    pytest.main(["-v", __file__])

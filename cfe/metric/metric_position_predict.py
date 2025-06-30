import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from cfe.util.expand_matrix import expand_matrix


def calculate_position_predict_fadata(
    fadata,
    prediction=None,
    metrics=["rf_mse", "rf_rsq", "rf_nmse", "lm_mse", "lm_rsq", "lm_nmse"],
    model_name=None
):
    """
    Compute metrics related to the prediction of cell positions for FateAnnData.

    Parameters:
        fadata : FateAnnData
            包含轨迹数据的 FateAnnData 对象，其轨迹信息存储在 fadata.uns["cfe"]["trajectory_history_dict"] 中。
        prediction : FateAnnData or None
            一个预测轨迹的 FateAnnData 对象，格式与 fadata 类似。
        metrics : list of str, optional
            指标列表，可选 "rf_mse", "rf_rsq", "rf_nmse", "lm_mse", "lm_rsq", "lm_nmse" 中的一项或多项。
        model_name : str or None
            指定使用哪一个轨迹模型。如果为 None，则默认为 fadata.model_name。

    Returns:
        dict: 返回一个包含 summary 键的字典，其中 summary 下保存总体各项指标，
              同时可能还包含各里程碑的详细指标（例如 rf_mses、lm_rsqs）。
    """
    # 1. 获取细胞ID列表
    cell_ids = fadata.obs.index.tolist()
    output = {"summary": {}}

    # 2. 根据模型名称获取 MilestoneWrapper 对象
    if model_name is None:
        model_name = fadata.model_name
    # 这里假设 fadata.uns["cfe"]["trajectory_history_dict"] 结构中 key 为 model_name
    milestone_wrapper = fadata.uns["cfe"]["trajectory_history_dict"].get(model_name, {}).get("milestone_wrapper", None)
    if milestone_wrapper is None:
        raise ValueError(f"No milestone_wrapper found for model: {model_name}")

    # 3. 从 MilestoneWrapper 中获取金标准的里程碑百分比数据
    # 假设 milestone_wrapper.milestone_percentages 是个 DataFrame，包含 "cell_id", "milestone_id", "percentage"
    gold_mp = milestone_wrapper.milestone_percentages
    gold_milenet_m = pd.pivot_table(
        gold_mp,
        index="cell_id",
        columns="milestone_id",
        values="percentage",
        fill_value=0
    )
    # 使用 expand_matrix 保证行顺序与 cell_ids 对齐
    gold_milenet_m = expand_matrix(gold_milenet_m, rownames=cell_ids)

    # 4. 计算 baseline_mse：对每个里程碑列计算各值与均值差的平方均值，再取所有列的平均
    baseline_mse = np.mean([
        np.mean((gold_milenet_m[col] - gold_milenet_m[col].mean())**2)
        for col in gold_milenet_m.columns
    ])

    # 5. 如果 prediction 有效，则计算预测指标
    if (prediction is not None and
        len(pd.unique(prediction.milestone_wrapper.milestone_percentages["cell_id"])) >= 3):
        # 从 prediction 的 MilestoneWrapper 中提取预测的里程碑百分比数据
        pred_mp = prediction.milestone_wrapper.milestone_percentages
        pred_milenet_m = pd.pivot_table(
            pred_mp,
            index="cell_id",
            columns="milestone_id",
            values="percentage",
            fill_value=0
        )
        pred_milenet_m = expand_matrix(pred_milenet_m, rownames=cell_ids)
        # 仅保留标准差大于 0 的列
        cols = [col for col in pred_milenet_m.columns if pred_milenet_m[col].std() > 0]
        pred_milenet_m = pred_milenet_m[cols]

        # 5.1 使用随机森林模型计算指标
        if any(metric in metrics for metric in ["rf_mse", "rf_rsq", "rf_nmse"]):
            rf_mses = {}
            rf_rsqs = {}
            for col in gold_milenet_m.columns:
                # 构造数据：目标列 “PREDICT”为金标准对应的百分比，预测变量为 pred_milenet_m 中的所有列
                target = gold_milenet_m[[col]].rename(columns={col: "PREDICT"})
                data = pd.concat([target, pred_milenet_m], axis=1)
                rf = RandomForestRegressor(n_estimators=5000, n_jobs=1, random_state=42)
                rf.fit(data.drop("PREDICT", axis=1), data["PREDICT"])
                preds = rf.predict(data.drop("PREDICT", axis=1))
                mse = mean_squared_error(data["PREDICT"], preds)
                rf_mses[col] = mse
                rsq = rf.score(data.drop("PREDICT", axis=1), data["PREDICT"])
                if np.isnan(rsq):
                    rsq = 1
                rf_rsqs[col] = rsq
            output["rf_mses"] = rf_mses
            output["summary"]["rf_mse"] = np.mean(list(rf_mses.values()))
            output["rf_rsqs"] = rf_rsqs
            output["summary"]["rf_rsq"] = max(np.mean(list(rf_rsqs.values())), 0)
            output["summary"]["rf_nmse"] = max(1 - output["summary"]["rf_mse"] / baseline_mse, 0)

        # 5.2 使用线性回归模型计算指标
        if any(metric in metrics for metric in ["lm_mse", "lm_rsq", "lm_nmse"]):
            lm_mses = []
            lm_rsqs = {}
            for col in gold_milenet_m.columns:
                target = gold_milenet_m[[col]].rename(columns={col: "PREDICT"})
                data = pd.concat([target, pred_milenet_m], axis=1)
                lr = LinearRegression()
                lr.fit(data.drop("PREDICT", axis=1), data["PREDICT"])
                preds = lr.predict(data.drop("PREDICT", axis=1))
                mse = np.mean((data["PREDICT"] - preds)**2)
                lm_mses.append(mse)
                rsq = lr.score(data.drop("PREDICT", axis=1), data["PREDICT"])
                lm_rsqs[col] = rsq if not np.isnan(rsq) else 1
            output["summary"]["lm_mse"] = np.mean(lm_mses)
            output["lm_rsqs"] = lm_rsqs
            output["summary"]["lm_rsq"] = max(np.mean(list(lm_rsqs.values())), 0)
            output["summary"]["lm_nmse"] = max(1 - output["summary"]["lm_mse"] / baseline_mse, 0)
    else:
        # 若预测数据不足，则返回 baseline 指标
        output["summary"] = {
            "rf_mse": baseline_mse,
            "rf_nmse": 0,
            "rf_rsq": 0,
            "lm_mse": baseline_mse,
            "lm_rsq": 0,
            "lm_nmse": 0
        }

    return output

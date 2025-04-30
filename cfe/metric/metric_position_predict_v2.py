import numpy as np
import pandas as pd
from typing import List, Dict, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from cfe.util.expand_matrix import expand_matrix
from cfe.data import FateAnnData

def calculate_position_predict(
    fadata: FateAnnData,
    ref_model: str = "ref",
    pred_model: str = "default",
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    Compute cell-position–prediction metrics (RF and LM) by comparing two trajectories
    stored inside the same FateAnnData.

    Both models must have been added by add_trajectory() and then add_waypoints(),
    so that each has a MilestoneWrapper in
      fadata.uns["cfe"]["trajectory_history_dict"][model_name]["milestone_wrapper"].

    Args:
        fadata: FateAnnData containing >=2 trajectories.
        ref_model: key for the reference trajectory.
        pred_model: key for the predicted trajectory.
        metrics: list of metrics to compute; default is all of
            ["rf_mse","rf_rsq","rf_nmse","lm_mse","lm_rsq","lm_nmse"]

    Returns:
        A dict with
          - "summary": {metric_name: float, ...}
          - per-milestone dicts "rf_mses", "rf_rsqs", "lm_rsqs" when computed.
    """
    if metrics is None:
        metrics = ["rf_mse","rf_rsq","rf_nmse","lm_mse","lm_rsq","lm_nmse"]

    # 1. 拉取两个 MilestoneWrapper
    hist = fadata.uns.get("cfe", {}).get("trajectory_history_dict", {})
    ref_w = hist.get(ref_model, {}).get("milestone_wrapper")
    pred_w= hist.get(pred_model, {}).get("milestone_wrapper")
    if ref_w is None:
        raise ValueError(f"Reference model '{ref_model}' has no milestone_wrapper")
    if pred_w is None:
        raise ValueError(f"Prediction model '{pred_model}' has no milestone_wrapper")

    # 2. 构建金标准 vs 预测的百分比矩阵 (cells × milestones)
    cells = list(fadata.obs.index)
    def _matrix_from_wrapper(w):
        mp = w.milestone_percentages
        mat = pd.pivot_table(
            mp, index="cell_id", columns="milestone_id", values="percentage", fill_value=0
        )
        return expand_matrix(mat, rownames=cells)

    gold_m = _matrix_from_wrapper(ref_w)
    pred_m = _matrix_from_wrapper(pred_w)

    # 3. baseline MSE (每个里程碑预测其自身平均值)
    baseline_mse = np.mean([
        ((gold_m[col] - gold_m[col].mean()) ** 2).mean()
        for col in gold_m.columns
    ])

    out: Dict[str, Any] = {"summary": {}}

    # 如果预测样本过少，直接返回 baseline
    if pred_w.milestone_percentages["cell_id"].nunique() < 3:
        out["summary"].update({
            "rf_mse": baseline_mse, "rf_rsq": 0.0, "rf_nmse": 0.0,
            "lm_mse": baseline_mse, "lm_rsq": 0.0, "lm_nmse": 0.0,
        })
        return out

    # 为了稳定，只保留在 pred_m 中方差>0 的列
    valid_cols = [c for c in pred_m.columns if pred_m[c].std() > 0]
    pred_m = pred_m[valid_cols]

    # 4. 随机森林部分
    if any(m in metrics for m in ("rf_mse","rf_rsq","rf_nmse")):
        rf_mses: Dict[str, float] = {}
        rf_rsqs: Dict[str, float] = {}
        for col in gold_m.columns:
            # DataFrame: 目标列为 gold 中当前 col，特征为 pred_m
            df = pd.concat([
                gold_m[[col]].rename(columns={col: "target"}),
                pred_m
            ], axis=1)
            X = df.drop("target", axis=1)
            y = df["target"]

            rf = RandomForestRegressor(n_estimators=5000, random_state=42, n_jobs=1)
            rf.fit(X, y)
            preds = rf.predict(X)

            mse = mean_squared_error(y, preds)
            rsq = rf.score(X, y)
            rf_mses[col] = mse
            rf_rsqs[col] = 1.0 if np.isnan(rsq) else rsq

        out["rf_mses"] = rf_mses
        out["rf_rsqs"] = rf_rsqs
        out["summary"]["rf_mse"] = np.mean(list(rf_mses.values()))
        out["summary"]["rf_rsq"] = max(0.0, np.mean(list(rf_rsqs.values())))
        out["summary"]["rf_nmse"] = max(0.0, 1 - out["summary"]["rf_mse"] / baseline_mse)

    # 5. 线性回归部分
    if any(m in metrics for m in ("lm_mse","lm_rsq","lm_nmse")):
        lm_mses: List[float] = []
        lm_rsqs: Dict[str, float] = {}
        for col in gold_m.columns:
            df = pd.concat([
                gold_m[[col]].rename(columns={col: "target"}),
                pred_m
            ], axis=1)
            X = df.drop("target", axis=1)
            y = df["target"]

            lr = LinearRegression()
            lr.fit(X, y)
            preds = lr.predict(X)

            mse = mean_squared_error(y, preds)
            rsq = lr.score(X, y)
            lm_mses.append(mse)
            lm_rsqs[col] = 1.0 if np.isnan(rsq) else rsq

        out["lm_rsqs"] = lm_rsqs
        out["summary"]["lm_mse"] = np.mean(lm_mses)
        out["summary"]["lm_rsq"] = max(0.0, np.mean(list(lm_rsqs.values())))
        out["summary"]["lm_nmse"] = max(0.0, 1 - out["summary"]["lm_mse"] / baseline_mse)

    return out

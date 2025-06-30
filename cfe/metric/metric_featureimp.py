import inspect

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import ks_2samp, ranksums
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from cfe.util.expand_matrix import expand_matrix


def get_expression(trajectory, expression_source="expression"):
    """
    获取表达矩阵的函数，兼容 FateAnnData 对象和 dict 格式。
    对于 FateAnnData 对象：
      - 如果 expression_source 为字符串，优先尝试从 obsm 中获取对应键；若不存在，则使用 X。
    对于 dict 对象，则直接返回 trajectory[expression_source]（假定已存储为矩阵）。
    """
    if hasattr(trajectory, "obs"):
        # FateAnnData 情形
        if isinstance(expression_source, str):
            if hasattr(trajectory, "obsm") and expression_source in trajectory.obsm:
                expr = trajectory.obsm[expression_source]
            else:
                expr = trajectory.X
        else:
            expr = expression_source
        if issparse(expr):
            expr = pd.DataFrame(expr.toarray(), index=trajectory.obs.index)
        elif not isinstance(expr, pd.DataFrame):
            expr = pd.DataFrame(expr, index=trajectory.obs.index)
        return expr
    else:
        expr = trajectory.get(expression_source)
        if issparse(expr):
            expr = pd.DataFrame(expr.toarray())
        elif not isinstance(expr, pd.DataFrame):
            expr = pd.DataFrame(expr)
        return expr

def is_wrapper_with_trajectory(trajectory):
    """
    判断轨迹对象是否包含轨迹信息。
    对于 FateAnnData 对象：
      - 如果对象具有 milestone_wrapper 属性且不为 None，返回 True；
      - 或者如果对象有 is_wrapped_with_trajectory 标志并为 True也返回 True。
    对于 dict 格式，则检查键 "pydynwrap:with_trajectory"。
    """
    if hasattr(trajectory, "obs"):
        if hasattr(trajectory, "is_wrapped_with_trajectory"):
            return bool(trajectory.is_wrapped_with_trajectory)
        else:
            return hasattr(trajectory, "milestone_wrapper") and trajectory.milestone_wrapper is not None
    else:
        return trajectory.get("pydynwrap:with_trajectory", False)

def apply_function_params(params: dict, nrow, ncol):
    """
    遍历字典中的每个参数，如果参数是可调用对象且其签名正好要求 nrow 和 ncol，
    则调用该函数，传入 nrow 与 ncol，并用返回值替换原参数值。
    """
    new_params = {}
    for key, value in params.items():
        if callable(value):
            sig = inspect.signature(value)
            param_names = list(sig.parameters.keys())
            if len(param_names) == 2 and set(param_names) == {"nrow", "ncol"}:
                new_params[key] = value(nrow=nrow, ncol=ncol)
            else:
                new_params[key] = value
        else:
            new_params[key] = value
    return new_params

def fi_ranger_rf(num_trees, mtry, sample_fraction, min_node_size, **kwargs):
    """
    构造一个基于 scikit-learn RandomForestRegressor 的特征重要性函数，
    用于计算连续目标（如里程碑百分比）的特征重要性。
    模拟 R 中 ranger 包的行为。
    """
    default_params = {
        "n_jobs": 1,
        "min_samples_leaf": min_node_size,
        "importance": "impurity",
        "write_forest": False
    }
    params = {**default_params, **kwargs}

    def fi_function(X, y, verbose=False):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        nrow, ncol = X.shape

        max_features = mtry(nrow=nrow, ncol=ncol) if callable(mtry) else mtry
        fraction = sample_fraction(nrow=nrow, ncol=ncol) if callable(sample_fraction) else sample_fraction
        max_samples = fraction if fraction < 1 else None

        data = X.copy()
        data.insert(0, "PREDICT", y)

        rf = RandomForestRegressor(
            n_estimators=num_trees,
            max_features=max_features,
            min_samples_leaf=params.get("min_samples_leaf"),
            n_jobs=params.get("n_jobs"),
            max_samples=max_samples,
            bootstrap=True if max_samples is not None else False,
            random_state=42
        )
        if verbose:
            print("Training RandomForestRegressor with parameters:", rf.get_params())
        rf.fit(data.drop("PREDICT", axis=1), data["PREDICT"])
        importance = rf.feature_importances_
        return dict(zip(X.columns, importance))

    return {"fun": fi_function}

def fi_ranger_rf_lite(num_trees=2000, num_variables_per_split=50, num_samples_per_tree=250, min_node_size=20, **kwargs):
    """
    轻量级版本，封装了 fi_ranger_rf 的参数，并提供默认值。
    """
    def mtry(nrow, ncol):
        return min(num_variables_per_split, ncol)
    def sample_fraction(nrow, ncol):
        return min(num_samples_per_tree / nrow, 1)
    return fi_ranger_rf(num_trees, mtry, sample_fraction, min_node_size, **kwargs)

def fi_caret(caret_method, **kwargs):
    """
    模拟 R 中 caret 包接口的特征重要性函数（仅支持 'rf'），使用分类模型，仅作示例。
    """
    if caret_method != "rf":
        raise ValueError("Invalid method. Only 'rf' is supported in this demo.")

    def fi_function(X, y, verbose=False):
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(random_state=42, **kwargs)
        rf.fit(X, y)
        importance = rf.feature_importances_
        if isinstance(X, pd.DataFrame):
            return dict(zip(X.columns, importance))
        else:
            return dict(enumerate(importance))

    return {"fun": fi_function}

def fi_ranger_rf_tiny(num_trees=100, num_variables_per_split=50, num_samples_per_tree=250, min_node_size=20, **kwargs):
    """
    fi_ranger_rf 的轻量级小型版本。
    """
    def mtry(nrow, ncol):
        return min(num_variables_per_split, ncol)
    def sample_fraction(nrow, ncol):
        return min(num_samples_per_tree / nrow, 1)
    return fi_ranger_rf(num_trees, mtry, sample_fraction, min_node_size, **kwargs)

def calculate_feature_importances(X, Y, fi_method=fi_ranger_rf_lite(), verbose=False):
    """
    计算特征重要性，返回一个 DataFrame，包含 predictor_id, feature_id 和 importance 三列。
    """
    if not isinstance(Y, pd.DataFrame):
        if issparse(Y):
            Y = pd.DataFrame(Y.toarray())
        else:
            Y = pd.DataFrame(Y)
    if issparse(X):
        X = pd.DataFrame(X.toarray())
    elif not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    result_list = []
    for predictor in Y.columns:
        if verbose:
            print(f"Calculating importance for predictor '{predictor}'...")
        y = Y[predictor]
        if y.dtype == object or y.dtype == bool:
            y = pd.Categorical(y)
        if len(y.unique()) == 1:
            importance_dict = {feat: 0 for feat in X.columns}
        else:
            importance_dict = fi_method["fun"](X, y, verbose=verbose)
        df = pd.DataFrame({
            "predictor_id": predictor,
            "feature_id": list(importance_dict.keys()),
            "importance": list(importance_dict.values())
        })
        result_list.append(df)
    result_df = pd.concat(result_list, ignore_index=True)
    result_df = result_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return result_df

def calculate_milestone_feature_importance(trajectory,
        expression_source="expression",
        milestones_oi=None,
        fi_method=fi_ranger_rf_lite(),
        verbose=False):
    """
    计算每个里程碑的特征重要性，返回 DataFrame 包含三列：milestone_id, feature_id, importance。
    """
    if hasattr(trajectory, "obs"):
        expression = get_expression(trajectory, expression_source)
        cell_ids = trajectory.obs.index.tolist()
        milestone_percentages = trajectory.milestone_wrapper.milestone_percentages
        milestone_ids = getattr(trajectory.milestone_wrapper, "id_list",
                                  sorted(milestone_percentages["milestone_id"].unique()))
    else:
        expression = get_expression(trajectory, expression_source)
        cell_ids = trajectory["cell_ids"]
        milestone_percentages = trajectory["milestone_percentages"]
        milestone_ids = trajectory.get("milestone_ids", sorted(milestone_percentages["milestone_id"].unique()))

    if not set(cell_ids).issubset(set(expression.index)):
        raise ValueError("Not all cell_ids in trajectory are present in the expression matrix.")
    if len(cell_ids) < 3:
        raise ValueError("Need 3 or more cells in a trajectory to calculate feature importance.")

    if milestones_oi is None:
        milestones_oi = milestone_ids

    mp_filtered = milestone_percentages[milestone_percentages["milestone_id"].isin(milestones_oi)]
    milenet_m = mp_filtered.pivot_table(index="cell_id", columns="milestone_id",
                                          values="percentage", fill_value=0)
    milenet_m = expand_matrix(milenet_m, rownames=cell_ids)

    imp_df = calculate_feature_importances(expression, milenet_m, fi_method=fi_method, verbose=verbose)
    imp_df = imp_df.rename(columns={"predictor_id": "milestone_id"})
    return imp_df

def calculate_overall_feature_importance(trajectory,
        expression_source="expression",
        fi_method=fi_ranger_rf_lite(),
        verbose=False):
    """
    计算整体特征重要性（跨里程碑），返回 DataFrame 包含 feature_id 与 importance。
    """
    milestone_imp = calculate_milestone_feature_importance(
        trajectory,
        expression_source=expression_source,
        fi_method=fi_method,
        verbose=verbose
    )
    overall = milestone_imp.groupby("feature_id", as_index=False)["importance"].mean()
    overall = overall.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return overall

def _calculate_featureimp_cor(dataset_imp, pred_imp):
    """
    计算两个整体特征重要性 DataFrame 之间的相关性和加权相关性。
    返回 dict，如 {"featureimp_cor": ..., "featureimp_wcor": ...}
    """
    join = pd.merge(
        dataset_imp.rename(columns={"importance": "dataset_imp"}),
        pred_imp.rename(columns={"importance": "pred_imp"}),
        on="feature_id",
        how="outer"
    )
    join["dataset_imp"] = join["dataset_imp"].fillna(0)
    join["pred_imp"] = join["pred_imp"].fillna(0)

    if join["dataset_imp"].std() == 0 or join["pred_imp"].std() == 0:
        return {"featureimp_cor": 0, "featureimp_wcor": 0}
    else:
        corr = np.corrcoef(join["dataset_imp"], join["pred_imp"])[0,1]
        corr = max(corr, 0)
        weights = join["dataset_imp"].values
        if np.sum(weights) == 0:
            wcor = 0
        else:
            weights = weights / np.sum(weights)
            mean_x = np.average(join["dataset_imp"], weights=weights)
            mean_y = np.average(join["pred_imp"], weights=weights)
            cov = np.average((join["dataset_imp"] - mean_x) * (join["pred_imp"] - mean_y), weights=weights)
            var_x = np.average((join["dataset_imp"] - mean_x)**2, weights=weights)
            var_y = np.average((join["pred_imp"] - mean_y)**2, weights=weights)
            if var_x * var_y > 0:
                wcor = cov / np.sqrt(var_x * var_y)
            else:
                wcor = 0
            wcor = max(wcor, 0)
        return {"featureimp_cor": corr, "featureimp_wcor": wcor}

def calculate_featureimp_cor(dataset, prediction, expression_source=None, fi_method=fi_ranger_rf_lite()):
    """
    比较两个轨迹计算得到的整体特征重要性之间的相关性，返回 dict。
    """
    if prediction is not None:
        if hasattr(prediction, "milestone_wrapper"):
            pred_cell_count = len(prediction.obs.index.tolist())
        else:
            pred_cell_count = len(pd.unique(prediction["milestone_percentages"]["cell_id"]))
    else:
        pred_cell_count = 0

    if prediction is not None and pred_cell_count >= 3:
        dataset_imp = calculate_overall_feature_importance(
            trajectory=dataset,
            expression_source=expression_source,
            fi_method=fi_method
        )
        pred_imp = calculate_overall_feature_importance(
            trajectory=prediction,
            expression_source=expression_source,
            fi_method=fi_method
        )
        return _calculate_featureimp_cor(dataset_imp, pred_imp)
    else:
        return {"featureimp_cor": 0, "featureimp_wcor": 0}

def calculate_featureimp_enrichment(dataset, prediction, expression_source=None, fi_method=fi_ranger_rf_lite()):
    """
    比较预测轨迹中特征的重要性富集情况，返回 dict，如 {"featureimp_ks": ks_pvalue, "featureimp_wilcox": 1 - wilcox_pvalue}
    """
    try:
        if prediction is not None:
            if hasattr(prediction, "milestone_wrapper"):
                pred_cell_count = len(prediction.obs.index.tolist())
            else:
                pred_cell_count = len(pd.unique(prediction["milestone_percentages"]["cell_id"]))
        else:
            pred_cell_count = 0

        if prediction is not None and pred_cell_count >= 3:
            pred_imp = calculate_overall_feature_importance(
                trajectory=prediction,
                expression_source=expression_source,
                fi_method=fi_method
            )
            if hasattr(dataset, "prior_information"):
                dataset_features = dataset.prior_information.get("features_id", [])
            else:
                dataset_features = dataset.get("prior_information", {}).get("features_id", [])
            sel = pred_imp.loc[pred_imp["feature_id"].isin(dataset_features), "importance"]
            notsel = pred_imp.loc[~pred_imp["feature_id"].isin(dataset_features), "importance"]

            if len(notsel) > 2:
                ks = ks_2samp(sel, notsel, alternative="greater")
                wilcox = ranksums(sel, notsel, alternative="greater")
                return {
                    "featureimp_ks": ks.pvalue,
                    "featureimp_wilcox": 1 - wilcox.pvalue
                }
            else:
                return {"featureimp_ks": 1, "featureimp_wilcox": 1}
        else:
            # 修改此处，使键名与其他情况一致，返回 "featureimp_wilcox" 而不是 "featureimp_wcor"
            return {"featureimp_ks": 0, "featureimp_wilcox": 0}
    except Exception as e:
        print("featureimp_enrichment errored! check reason!", e)
        return {"featureimp_ks": 0, "featureimp_wilcox": 0}

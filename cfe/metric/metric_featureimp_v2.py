import inspect
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import ks_2samp, ranksums
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Any, Callable, Dict, Optional

from cfe.data import FateAnnData
from cfe.util.expand_matrix import expand_matrix


def get_expression(trajectory: Any, expression_source: Any = "expression") -> pd.DataFrame:
    """
    获取表达矩阵，兼容 FateAnnData 和 dict.
    对于 FateAnnData：优先取 obsm[expression_source]，否则使用 X；
    对于 dict，直接取 trajectory[expression_source].
    返回 DataFrame，index 对齐细胞 ID.
    """
    if hasattr(trajectory, "obs"):
        # FateAnnData
        if isinstance(expression_source, str) and hasattr(trajectory, 'obsm') and expression_source in trajectory.obsm:
            expr = trajectory.obsm[expression_source]
        else:
            expr = trajectory.X
        if issparse(expr):
            expr = pd.DataFrame(expr.toarray(), index=trajectory.obs.index)
        elif not isinstance(expr, pd.DataFrame):
            expr = pd.DataFrame(expr, index=trajectory.obs.index)
        return expr
    else:
        # dict-like
        expr = trajectory.get(expression_source)
        if issparse(expr):
            expr = pd.DataFrame(expr.toarray())
        elif not isinstance(expr, pd.DataFrame):
            expr = pd.DataFrame(expr)
        return expr


def is_wrapper_with_trajectory(trajectory: Any) -> bool:
    """
    判断对象是否已包装轨迹。FateAnnData 检查 is_wrapped_with_trajectory 或 milestone_wrapper。dict 检查 "pydynwrap:with_trajectory"。
    """
    if hasattr(trajectory, 'obs'):
        if hasattr(trajectory, 'is_wrapped_with_trajectory'):
            return bool(trajectory.is_wrapped_with_trajectory)
        return hasattr(trajectory, 'milestone_wrapper') and trajectory.milestone_wrapper is not None
    else:
        return trajectory.get("pydynwrap:with_trajectory", False)


def apply_function_params(params: Dict[str, Any], nrow: int, ncol: int) -> Dict[str, Any]:
    """
    遍历 params 字典，如果值是 signature(nrow,ncol) 的函数，则调用获得新的值。
    """
    new_params: Dict[str, Any] = {}
    for k, v in params.items():
        if callable(v):
            sig = inspect.signature(v)
            names = list(sig.parameters.keys())
            if set(names) == {'nrow', 'ncol'}:
                new_params[k] = v(nrow=nrow, ncol=ncol)
            else:
                new_params[k] = v
        else:
            new_params[k] = v
    return new_params


def fi_ranger_rf(
    num_trees: int,
    mtry: Callable[[int,int], int],
    sample_fraction: Callable[[int,int], float],
    min_node_size: int,
    **kwargs
) -> Dict[str, Callable]:
    """
    基于 RandomForestRegressor 的特征重要性函数，模拟 R ranger.
    返回 {'fun': fi_function}。
    """
    def fi_function(X, y, verbose: bool = False) -> Dict[Any, float]:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        nrow, ncol = X.shape
        max_features = mtry(nrow=nrow, ncol=ncol)
        fraction = sample_fraction(nrow=nrow, ncol=ncol)
        max_samples = fraction if fraction < 1 else None

        df = X.copy()
        df.insert(0, 'target', y)

        rf = RandomForestRegressor(
            n_estimators=num_trees,
            max_features=max_features,
            min_samples_leaf=min_node_size,
            max_samples=max_samples,
            bootstrap=(max_samples is not None),
            random_state=42,
            n_jobs=1,
            **{k: v for k, v in kwargs.items() if k in ['criterion', 'min_weight_fraction_leaf']}
        )
        if verbose:
            print("RF params:", rf.get_params())
        rf.fit(df.drop('target', axis=1), df['target'])
        return dict(zip(X.columns, rf.feature_importances_))

    return {'fun': fi_function}


def fi_ranger_rf_lite(
    num_trees: int = 2000,
    num_variables_per_split: int = 50,
    num_samples_per_tree: int = 250,
    min_node_size: int = 20,
    **kwargs
) -> Dict[str, Callable]:
    """
    轻量版 Ranger RF。默认参数封装。
    """
    def mtry(nrow, ncol):
        return min(num_variables_per_split, ncol)
    def sample_fraction(nrow, ncol):
        return min(num_samples_per_tree / nrow, 1)
    return fi_ranger_rf(num_trees, mtry, sample_fraction, min_node_size, **kwargs)


def fi_caret(caret_method: str = 'rf', **kwargs) -> Dict[str, Callable]:
    """
    模拟 caret 接口，仅支持 'rf'。返回 {'fun': fi_function}。
    """
    if caret_method != 'rf':
        raise ValueError("Only 'rf' supported in fi_caret")
    def fi_function(X, y, verbose: bool = False):
        rf = RandomForestClassifier(random_state=42, **kwargs)
        rf.fit(X, y)
        imp = rf.feature_importances_
        cols = X.columns if isinstance(X, pd.DataFrame) else range(len(imp))
        return dict(zip(cols, imp))
    return {'fun': fi_function}


def fi_ranger_rf_tiny(
    num_trees: int = 100,
    num_variables_per_split: int = 50,
    num_samples_per_tree: int = 250,
    min_node_size: int = 20,
    **kwargs
) -> Dict[str, Callable]:
    """小型版 Ranger RF。"""
    return fi_ranger_rf_lite(num_trees, num_variables_per_split, num_samples_per_tree, min_node_size, **kwargs)


def calculate_feature_importances(
    X: Any,
    Y: Any,
    fi_method: Dict[str, Callable] = fi_ranger_rf_lite(),
    verbose: bool = False
) -> pd.DataFrame:
    """
    对每个预测 (Y 列) 计算 predictor -> feature 重要性。
    返回 DataFrame [predictor_id, feature_id, importance].
    """
    # 转 DataFrame
    if issparse(Y := Y if not hasattr(Y, 'values') else Y):
        Y = pd.DataFrame(Y.toarray())
    elif not isinstance(Y, pd.DataFrame):
        Y = pd.DataFrame(Y)
    if issparse(X := X if not hasattr(X, 'values') else X):
        X = pd.DataFrame(X.toarray())
    elif not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    results = []
    for pred in Y.columns:
        if verbose:
            print(f"Calculating importance for '{pred}'")
        y = Y[pred]
        if y.dtype == object or y.dtype == bool:
            y = pd.Categorical(y)
        if y.nunique() == 1:
            imp = {f: 0.0 for f in X.columns}
        else:
            imp = fi_method['fun'](X, y, verbose)
        df = pd.DataFrame({
            'predictor_id': pred,
            'feature_id': list(imp.keys()),
            'importance': list(imp.values())
        })
        results.append(df)
    out = pd.concat(results, ignore_index=True)
    return out.sort_values('importance', ascending=False).reset_index(drop=True)


def calculate_milestone_feature_importance(
    trajectory: Any,
    expression_source: Any = 'expression',
    milestones_oi: Optional[list] = None,
    fi_method: Dict[str, Callable] = fi_ranger_rf_lite(),
    verbose: bool = False
) -> pd.DataFrame:
    """
    对每个里程碑计算特征重要性，输出 [milestone_id, feature_id, importance].
    """
    if hasattr(trajectory, 'obs'):
        expr = get_expression(trajectory, expression_source)
        cells = trajectory.obs.index.tolist()
        mp = trajectory.milestone_wrapper.milestone_percentages
        all_ms = getattr(trajectory.milestone_wrapper, 'id_list', sorted(mp['milestone_id'].unique()))
    else:
        expr = get_expression(trajectory, expression_source)
        cells = trajectory['cell_ids']
        mp = trajectory['milestone_percentages']
        all_ms = trajectory.get('milestone_ids', sorted(mp['milestone_id'].unique()))

    if not set(cells) <= set(expr.index):
        raise ValueError("Expression missing some cell IDs")
    if len(cells) < 3:
        raise ValueError("Need >=3 cells for feature importance")

    if milestones_oi is None:
        milestones_oi = all_ms

    mpf = mp[mp['milestone_id'].isin(milestones_oi)]
    mmat = mpf.pivot_table(index='cell_id', columns='milestone_id', values='percentage', fill_value=0)
    mmat = expand_matrix(mmat, rownames=cells)

    imp_df = calculate_feature_importances(expr, mmat, fi_method, verbose)
    return imp_df.rename(columns={'predictor_id': 'milestone_id'})


def calculate_overall_feature_importance(
    trajectory: Any,
    expression_source: Any = 'expression',
    fi_method: Dict[str, Callable] = fi_ranger_rf_lite(),
    verbose: bool = False
) -> pd.DataFrame:
    """
    跨里程碑的整体特征重要性，按 feature_id 平均。
    返回 [feature_id, importance].
    """
    milimp = calculate_milestone_feature_importance(trajectory, expression_source, None, fi_method, verbose)
    overall = milimp.groupby('feature_id', as_index=False)['importance'].mean()
    return overall.sort_values('importance', ascending=False).reset_index(drop=True)


def _calculate_featureimp_cor(dataset_imp: pd.DataFrame, pred_imp: pd.DataFrame) -> Dict[str, float]:
    """
    计算两个整体 feature importance 的 Pearson 及加权相关。
    返回 {'featureimp_cor', 'featureimp_wcor'}.
    """
    df = pd.merge(
        dataset_imp.rename(columns={'importance': 'dataset_imp'}),
        pred_imp.rename(columns={'importance': 'pred_imp'}),
        on='feature_id', how='outer'
    ).fillna(0)

    if df['dataset_imp'].std() == 0 or df['pred_imp'].std() == 0:
        return {'featureimp_cor': 0.0, 'featureimp_wcor': 0.0}

    corr = np.corrcoef(df['dataset_imp'], df['pred_imp'])[0,1]
    corr = max(corr, 0.0)

    w = df['dataset_imp'].values
    if w.sum() == 0:
        wcor = 0.0
    else:
        w = w / w.sum()
        mx = np.average(df['dataset_imp'], weights=w)
        my = np.average(df['pred_imp'], weights=w)
        cov = np.average((df['dataset_imp']-mx)*(df['pred_imp']-my), weights=w)
        vx = np.average((df['dataset_imp']-mx)**2, weights=w)
        vy = np.average((df['pred_imp']-my)**2, weights=w)
        wcor = cov/np.sqrt(vx*vy) if vx*vy>0 else 0.0
        wcor = max(wcor, 0.0)

    return {'featureimp_cor': corr, 'featureimp_wcor': wcor}


def calculate_featureimp_cor(
    fadata: FateAnnData,
    ref_model: str = 'ref',
    pred_model: str = 'default',
    expression_source: Optional[str] = None,
    fi_method: Dict[str, Callable] = None
) -> Dict[str, float]:
    """
    对比同一 FateAnnData 中两条模型的整体特征重要性相关性。
    """
    hist = fadata.uns.get('cfe', {}).get('trajectory_history_dict', {})
    if ref_model not in hist or pred_model not in hist:
        return {'featureimp_cor': 0.0, 'featureimp_wcor': 0.0}

    orig = fadata.model_name
    # 参考
    fadata.model_name = ref_model
    imp_ref = calculate_overall_feature_importance(fadata, expression_source, fi_method)
    # 预测
    fadata.model_name = pred_model
    imp_pred= calculate_overall_feature_importance(fadata, expression_source, fi_method)
    # 恢复
    fadata.model_name = orig

    return _calculate_featureimp_cor(imp_ref, imp_pred)


def calculate_featureimp_enrichment(
    fadata: FateAnnData,
    ref_model: str = 'ref',
    pred_model: str = 'default',
    expression_source: Optional[str] = None,
    fi_method: Dict[str, Callable] = None
) -> Dict[str, float]:
    """
    在预测模型的 importance 中，对参考模型 prior_information['features_id'] 的富集检验。
    返回 {'featureimp_ks', 'featureimp_wilcox'}.
    """
    hist = fadata.uns.get('cfe', {}).get('trajectory_history_dict', {})
    if pred_model not in hist:
        return {'featureimp_ks': 0.0, 'featureimp_wilcox': 0.0}

    orig = fadata.model_name
    fadata.model_name = pred_model
    imp_pred = calculate_overall_feature_importance(fadata, expression_source, fi_method)
    fadata.model_name = ref_model
    features = fadata.prior_information.get('features_id', [])
    fadata.model_name = orig

    sel = imp_pred.loc[imp_pred['feature_id'].isin(features), 'importance'].values
    notel = imp_pred.loc[~imp_pred['feature_id'].isin(features), 'importance'].values
    if len(notel) < 3 or sel.size == 0:
        return {'featureimp_ks': 0.0, 'featureimp_wilcox': 0.0}

    ks = ks_2samp(sel, notel, alternative='greater')
    wilc = ranksums(sel, notel, alternative='greater')
    return {'featureimp_ks': ks.pvalue, 'featureimp_wilcox': 1.0 - wilc.pvalue}

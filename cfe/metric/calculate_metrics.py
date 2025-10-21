from typing import Callable, Dict, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from cfe.data import FateAnnData
from cfe.metric._topology_metric.metric_flip import calculate_edge_flip
from cfe.metric._topology_metric.metric_him import calculate_him
from cfe.metric.cluster_metric import (
    calculate_mapping_branches,
    calculate_mapping_milestones,
)
from cfe.metric.metric_correlation import calculate_correlation
from cfe.metric.metric_featureimp import (  # fi_ranger_rf_tiny,
    calculate_featureimp_cor,
    fi_ranger_rf_lite,
)
from cfe.metric.metric_position_predict import calculate_position_predict

from .._logging import logger

# from cfe.metric.topology_metric import calc_isomorphic


def calculate_metrics(
    fadata: FateAnnData,
    now_model: Union[str, List[str]] = "all",
    ref_model: str = "ref",
    simplify: bool = True,
    metrics: Optional[List[str]] = None,
    expression_source: str = "expression",
    fi_method: Dict[str, Callable] = None,
) -> pd.DataFrame:
    """
    计算一组指标（严格使用你指定的指标名），比较 ref_model vs 多个预测模型。

    默认 metrics（严格按你给出的名字）:
      ["isomorphic","edge_flip","him","correlation","F1_branches","F1_milestones",
       "rf_mse","rf_nmse","rf_rsq","lm_nmse","lm_mse","lm_rsq","featureimp_cor","featureimp_wcor"]

    返回 DataFrame：index 为预测模型名，columns 对应 metrics（若某些指标无效则为 NaN）。

    """
    if fi_method is None:
        fi_method = fi_ranger_rf_lite()

    # 默认严格指标集合（按你要求）
    if metrics is None:
        metrics = [
            "isomorphic",
            "edge_flip",
            "him",
            "correlation",
            "F1_branches",
            "F1_milestones",
            "rf_mse",
            "rf_nmse",
            "rf_rsq",
            "lm_nmse",
            "lm_mse",
            "lm_rsq",
            "featureimp_cor",
            "featureimp_wcor",
        ]

    # 获得存在的模型名
    hist = fadata.uns.get("cfe", {}).get("trajectory_history_dict", {})
    # 检查参考模型是否存在
    if ref_model not in hist:
        raise ValueError(f"参考模型 '{ref_model}' 不在 fadata.uns['cfe']['trajectory_history_dict'] 中。可用模型：{list(hist.keys())}")

    available_models = list(hist.keys())

    # 构建预测模型名
    if isinstance(now_model, list):
        pred_models = now_model
    elif isinstance(now_model, str) and now_model == "all":
        pred_models = [m for m in available_models if m != ref_model]
    else:
        pred_models = [now_model]

    # 过滤掉不存在的模型
    pred_models = [m for m in pred_models if m in hist]
    if len(pred_models) == 0:
        return pd.DataFrame(columns=metrics, index=[])

    rows = []
    idx = []

    # 辅助函数：获取里程碑网络（简化或原始）
    def _get_milestone_network(model_name: str, do_simplify: bool):
        try:
            if do_simplify:
                # 简化后的里程碑网络
                mw_simpl = fadata.simplify_trajectory(model_name)
                return mw_simpl.milestone_network
            else:
                md = hist.get(model_name, {})
                mw = md.get("milestone_wrapper")
                if mw is None:
                    return None
                if isinstance(mw, dict):
                    # 可能是个 dict，取其中的 milestone_network
                    mn = mw.get("milestone_network")
                    if mn is None:
                        return None
                    return pd.DataFrame(mn)
                else:
                    return mw.milestone_network
        except Exception:
            return None

    for pred in pred_models:
        _featureimp_cache = None  # 缓存特征重要性结果，避免重复计算
        idx.append(pred)
        vals = {m: np.nan for m in metrics}  # 初始化所有 requested metrics 为 NaN

        # 获取两个里程碑网络
        net_ref = _get_milestone_network(ref_model, simplify)
        net_pred = _get_milestone_network(pred, simplify)

        # 对每个指定的指标名进行映射计算（捕获异常并保留 NaN）
        resource_usage = fadata.get_resource_usage(model_name=pred)  # extract time and memory usage
        vals["cpu"] = resource_usage.get("cpu", np.nan)
        vals["memory"] = resource_usage.get("memory", np.nan)
        vals["time"] = resource_usage.get("time", np.nan)

        for metric in metrics:
            try:
                # TODO: add linear pseudotime correlation, velocity correlation...
                # TODO:
                if metric == "pseudotime":
                    pass
                elif metric == "velocity":
                    pass
                elif metric == "isomorphic":
                    if net_ref is None or net_pred is None or net_ref.shape[0] == 0 or net_pred.shape[0] == 0:
                        vals["isomorphic"] = np.nan
                    else:
                        G_ref = nx.from_pandas_edgelist(
                            net_ref.rename(columns={"length": "weight"}), source="from", target="to", create_using=nx.Graph
                        )
                        G_pred = nx.from_pandas_edgelist(
                            net_pred.rename(columns={"length": "weight"}), source="from", target="to", create_using=nx.Graph
                        )
                        vals["isomorphic"] = 1.0 if nx.is_isomorphic(G_ref, G_pred) else 0.0

                elif metric == "edge_flip":
                    if net_ref is None or net_pred is None:
                        vals["edge_flip"] = np.nan
                    else:
                        # 网络做了提前的简化，这里的symplify参数强制为False（默认值）
                        vals["edge_flip"] = float(calculate_edge_flip(net_ref, net_pred, return_type="score"))

                elif metric == "him":
                    if net_ref is None or net_pred is None:
                        vals["him"] = np.nan
                    else:
                        # 网络做了提前的简化，这里的symplify参数强制为False（默认值）
                        vals["him"] = float(calculate_him(net_ref, net_pred))

                elif metric == "correlation":
                    # 取其返回 dict 中的 'correlation'
                    try:
                        cm = calculate_correlation(fadata, ref_model=ref_model, pred_model=pred)
                        vals["correlation"] = float(cm.get("correlation", np.nan))
                    except Exception:
                        vals["correlation"] = np.nan

                elif metric == "F1_milestones":
                    try:
                        mm = calculate_mapping_milestones(fadata, ref_model=ref_model, pred_model=pred, simplify=simplify)
                        # mm 返回 {'recovery_milestones', 'relevance_milestones', 'F1_milestones'}
                        vals["F1_milestones"] = float(mm.get("F1_milestones", np.nan))
                    except Exception:
                        vals["F1_milestones"] = np.nan

                elif metric == "F1_branches":
                    try:
                        mb = calculate_mapping_branches(fadata, ref_model=ref_model, pred_model=pred, simplify=simplify)
                        vals["F1_branches"] = float(mb.get("F1_branches", np.nan))
                    except Exception:
                        vals["F1_branches"] = np.nan

                elif metric in ("rf_mse", "rf_nmse", "rf_rsq", "lm_mse", "lm_rsq", "lm_nmse"):
                    # 用一次 calculate_position_predict 得到所有位置预测指标
                    try:
                        pp = calculate_position_predict(fadata, ref_model=ref_model, pred_model=pred)
                        summary = pp.get("summary", {})
                        # rf
                        if "rf_mse" in summary:
                            vals["rf_mse"] = float(summary.get("rf_mse", np.nan))
                        if "rf_rsq" in summary:
                            vals["rf_rsq"] = float(summary.get("rf_rsq", np.nan))
                        if "rf_nmse" in summary:
                            vals["rf_nmse"] = float(summary.get("rf_nmse", np.nan))
                        # lm
                        if "lm_mse" in summary:
                            vals["lm_mse"] = float(summary.get("lm_mse", np.nan))
                        if "lm_rsq" in summary:
                            vals["lm_rsq"] = float(summary.get("lm_rsq", np.nan))
                        if "lm_nmse" in summary:
                            vals["lm_nmse"] = float(summary.get("lm_nmse", np.nan))
                    except Exception:
                        # 保持这些键为 NaN（已初始化）
                        pass

                elif metric in ("featureimp_cor", "featureimp_wcor"):
                    try:
                        if _featureimp_cache is None:
                            _featureimp_cache = calculate_featureimp_cor(
                                fadata,
                                ref_model=ref_model,
                                pred_model=pred,
                                expression_source=expression_source,  # 根据你的数据改成实际的 key
                                fi_method=fi_method,  # 使用默认轻量 RF，或者传你自定义的 fi_method
                            )
                        # 可能返回 np.nan，需要容错转换
                        vals["featureimp_cor"] = float(_featureimp_cache.get("featureimp_cor", np.nan))
                        vals["featureimp_wcor"] = float(_featureimp_cache.get("featureimp_wcor", np.nan))
                    except Exception:
                        vals["featureimp_cor"] = np.nan
                        vals["featureimp_wcor"] = np.nan
                # else:
                #     # 未知严格指标名 —— 保持 NaN
                #     vals[metric] = np.nan

            # TODO: 这里的Exception和内部的Exception的含义不一样吗？
            except Exception as e:
                # 任何子计算失败，不抛出，留下 NaN
                logger.warning(f"metric '{metric}' calculation failed for trajectory '{ref_model}(ref)' vs '{pred}(pred)'")
                logger.warning(f"Exception: {e}")
                vals[metric] = np.nan

        rows.append(vals)

    # 构造 DataFrame，确保列按传入 metrics 顺序（如果某些列没有出现在 vals 中，就仍然以传入的 metrics 列显示并填 NaN）
    df = pd.DataFrame(rows, index=idx)
    # 确保列存在且顺序一致（如果某些 requested metric 列不存在就补上 NaN）
    for m in metrics:
        if m not in df.columns:
            df[m] = np.nan
    df = df[metrics]

    return df

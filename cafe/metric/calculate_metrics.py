from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .._logging import logger
from ..data import FateAnnData
from . import (
    metric_cluster,
    metric_correlation,
    metric_embedding,
    metric_featureimp,
    metric_position_predict,
    metric_pseudotime,
    metric_resource,
    metric_topology,
    metric_velocity,
)


# TODO: add docs
def calculate_metrics(
    fadata: FateAnnData,
    now_models: Union[str, List[str]] = "all",
    ref_model: str = "ref",
    simplify: bool = True,
    metrics: Optional[List[str]] = None,
    cluster_edges: Optional[List[tuple]] = None,  # for velocity metric
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
        fi_method = metric_featureimp.fi_ranger_rf_lite()

    if metrics is None:
        # default metrics
        metrics = [
            "euclidean_distance_pc",
            "geodesic_distance_pc",
            "pseudotime_correlation",
            "velocity_cbdir",
            "velocity_icvcoh",
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
            "time",
            "time_text",
            "memory",
            "memory_text",
        ]

    # check available methods, fiter invalid models
    available_models = fadata.get_all_model_name(parse=False)  # need original model name for metric calculation and matching because mhn code style
    if ref_model not in available_models:
        raise ValueError(f"ref model('{ref_model}') not in available models: {available_models}")

    if isinstance(now_models, list):
        pred_models = now_models
    elif isinstance(now_models, str) and now_models == "all":
        pred_models = [m for m in available_models if m != ref_model]
    else:
        pred_models = [now_models]

    pred_models = [m for m in pred_models if m in available_models]
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
                mw = fadata.get_milestone_wrapper(model_name)
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

    net_ref = _get_milestone_network(ref_model, simplify)  # common ref net

    if cluster_edges is None:
        logger.debug(f"cluster_edges not provided, trying to get from milestone network of trajectory('{ref_model}').")
        cluster_edges = fadata.get_milestone_wrapper(ref_model).milestone_network[["from", "to"]].values.tolist()

    if ("velocity_cbdir" in metrics) and (cluster_edges is None):
        logger.warning("velocity metric calculation need parameter 'cluster_edges', skip it.")
        metrics.remove("velocity_cbdir")

    if (("velocity_cbdir" in metrics) or ("velocity_icvcoh" in metrics)) and ("neighbors" not in fadata.uns):
        logger.info("neighbors not found in fadata.uns, computing neighbors for velocity metric calculation.")
        import scvelo as scv

        scv.pp.filter_and_normalize(fadata, n_top_genes=2000)
        scv.pp.moments(fadata, n_pcs=30, n_neighbors=30)
        # TODO: fix for reconstruct FateAnnData

    for pred in pred_models:
        idx.append(pred)
        vals = {m: np.nan for m in metrics}  # 初始化所有 requested metrics 为 NaN
        net_pred = _get_milestone_network(pred, simplify)

        # 缓存机制，避免重复计算
        _embedding_cache = None
        _velocity_cache = None
        _position_cache = None
        _featureimp_cache = None
        _resource_cache = None

        for metric in metrics:
            try:
                if metric in ("euclidean_distance_pc", "geodesic_distance_pc"):
                    # emebdding metric, need cache
                    _embedding_cache = metric_embedding.calculate_embedding_metric(fadata, pre_trajectory=False, post_trajectory=False)
                    val = _embedding_cache[metric]
                if metric == "pseudotime_correlation":
                    # pseudotim
                    val = metric_pseudotime.calculate_pseudotime_correlation(fadata, ref_model=ref_model, pred_model=pred)
                elif metric == "velocity_cbdir" or metric == "velocity_icvcoh":
                    # velocity, need cache
                    if _velocity_cache is None:
                        _velocity_cache = metric_velocity.calculate_velocity_metrics(
                            fadata, cluster_edges=cluster_edges, model_name=pred
                        )  # dont't need ref model
                    val = _velocity_cache[metric]
                elif metric == "isomorphic":
                    # topology-isomorphic metric
                    val = metric_topology.calculate_isomorphic(net_ref, net_pred)
                elif metric == "edge_flip":
                    # topology-edge_flip metric
                    val = metric_topology.calculate_edge_flip(net_ref, net_pred, simplify=False)
                elif metric == "him":
                    # topology-him metric
                    val = metric_topology.calculate_him(net_ref, net_pred, simplify=False)
                elif metric == "correlation":
                    # correlation metric
                    val = metric_correlation.calculate_correlation(fadata, ref_model=ref_model, pred_model=pred)
                elif metric == "F1_milestones":
                    # cluster-milestone metric
                    val = metric_cluster.calculate_mapping_milestones(fadata, ref_model=ref_model, pred_model=pred, simplify=simplify)
                elif metric == "F1_branches":
                    # cluster-branch metric
                    val = metric_cluster.calculate_mapping_branches(fadata, ref_model=ref_model, pred_model=pred, simplify=simplify)
                elif metric in ("rf_mse", "rf_nmse", "rf_rsq", "lm_mse", "lm_rsq", "lm_nmse"):
                    # position metric
                    if _position_cache is None:
                        _position_cache = metric_position_predict.calculate_position_predict(fadata, ref_model=ref_model, pred_model=pred)
                    val = _position_cache["summary"][metric]
                elif metric in ("featureimp_cor", "featureimp_wcor"):
                    # feature imp metric, need cache
                    if _featureimp_cache is None:
                        _featureimp_cache = metric_featureimp.calculate_featureimp_cor(
                            fadata,
                            ref_model=ref_model,
                            pred_model=pred,
                            expression_source=expression_source,  # 根据你的数据改成实际的 key
                            fi_method=fi_method,  # 使用默认轻量 RF，或者传你自定义的 fi_method
                        )
                    val = _featureimp_cache[metric]
                elif metric in ["time", "time_text", "memory", "memory_text"]:
                    # time metric, need cache
                    if _resource_cache is None:
                        _resource_cache = metric_resource.calculate_resource_usage(fadata, model_name=pred, format_text=True)
                    val = _resource_cache[metric]
                vals[metric] = val
            except Exception as e:
                logger.warning(f"metric '{metric}' calculation failed for trajectory '{ref_model}(ref)' vs '{pred}(pred)'")
                logger.warning(f"Exception: {e}")
                vals[metric] = np.nan
                # raise e # for debug with exception

        rows.append(vals)

    # 构造 DataFrame，确保列按传入 metrics 顺序（如果某些列没有出现在 vals 中，就仍然以传入的 metrics 列显示并填 NaN）
    df = pd.DataFrame(rows, index=idx)
    # 确保列存在且顺序一致（如果某些 requested metric 列不存在就补上 NaN）
    for m in metrics:
        if m not in df.columns:
            df[m] = np.nan
    df = df[metrics]

    return df

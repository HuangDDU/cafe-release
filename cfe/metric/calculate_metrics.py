# from .topology_metric import calc_isomorphic, calculate_edge_flip
# from .cluster_metric_v2 import calculate_mapping_branches, calculate_mapping_milestones
# from cfe.metric._topology_metric.metric_him import calculate_him
# from cfe.metric.metric_correlation_v2 import calculate_correlation
# from cfe.metric.metric_position_predict_v2 import calculate_position_predict
# from cfe.metric.metric_featureimp_v2 import  calculate_overall_feature_importance,calculate_featureimp_cor,calculate_featureimp_enrichment

# def calculate_metrics(
#         fadata,
#         now_model=None,
#         ref_model=None,
#         simplify=True,
#         metrics=["isomorphic", "edge_flip"]
# ):
#     summary_dict = {}

#     if simplify:
#         net1 = fadata.simplify_trajectory(ref_model)["milestone_network"]
#         net2 = fadata.simplify_trajectory(now_model)["milestone_network"]
#     else:
#         net1 = fadata.trajectory_history_dict[ref_model]["milestone_wrapper"]["milestone_network"]
#         net2 = fadata.trajectory_history_dict[now_model]["milestone_wrapper"]["milestone_network"]

#     #metric correlation
#     if "correlation" in metrics:
#         summary_dict["correlation"]=calculate_correlation()

#     # topology metric
#     if "isomorphic" in metrics:
#         summary_dict["isomorphic"] = calc_isomorphic(net1, net2)
#     if "edge_flip" in metrics:
#         summary_dict["edge_flip"] = calculate_edge_flip(net1, net2)
#     if "him" in metrics:
#         summary_dict["him"] = calculate_him(net1, net2)

#     #position_predict metric
#     if any(x in metrics for x in ["rf_mse", "rf_rsq", "rf_nmse", "lm_mse", "lm_rsq", "lm_nmse"]):
#         pass

#     #featureimp metric
#     if any(x in metrics for x in ["featureimp_cor", "featureimp_wcor"]):
#         pass

#     if any(x in metrics for x in ["featureimp_ks", "featureimp_wilcox"]):
#         pass

#     # cluster metric
#     if "F1_branch" in metrics:
#         summary_dict["F1_branch"] = calculate_mapping_branches()
#     if "F1_milestone" in metrics:
#         summary_dict["F1_milestone"] = calculate_mapping_milestones()

from cfe.metric._topology_metric.metric_him import calculate_him
from cfe.metric.metric_correlation_v2 import calculate_correlation
from cfe.metric.metric_featureimp_v2 import (  # calculate_overall_feature_importance,
    calculate_featureimp_cor,
    calculate_featureimp_enrichment,
)
from cfe.metric.metric_position_predict_v2 import calculate_position_predict

from .cluster_metric_v2 import calculate_mapping_branches, calculate_mapping_milestones

#     return summary_dict
from .topology_metric import calc_isomorphic, calculate_edge_flip


def calculate_metrics(fadata, now_model: str, ref_model: str, simplify: bool = True, metrics: list[str] = ["isomorphic", "edge_flip"]) -> dict:
    """
    汇总多种轨迹对比指标。

    Args:
        fadata: FateAnnData，包含多条模型的 trajectory_history_dict。
        now_model: 当前（预测）模型的 key。
        ref_model: 参考模型的 key。
        simplify: 是否先调用 simplify_trajectory 简化网络。
        metrics: 要计算的指标列表。

    Returns:
        dict: 各指标的标量结果。
    """
    summary: dict[str, float] = {}

    # 1. 拿到两个里程碑网络（简化或原始）
    if simplify:
        net1 = fadata.simplify_trajectory(ref_model)["milestone_network"]
        net2 = fadata.simplify_trajectory(now_model)["milestone_network"]
    else:
        net1 = fadata.trajectory_history_dict[ref_model]["milestone_wrapper"]["milestone_network"]
        net2 = fadata.trajectory_history_dict[now_model]["milestone_wrapper"]["milestone_network"]
    # if simplify:
    #     simp_ref = fadata.simplify_trajectory(ref_model)
    #     net1 = simp_ref.milestone_network
    #     simp_now = fadata.simplify_trajectory(now_model)
    #     net2 = simp_now.milestone_network
    # else:
    #     hist = fadata.trajectory_history_dict
    #     net1 = hist[ref_model]["milestone_wrapper"].milestone_network
    #     net2 = hist[now_model]["milestone_wrapper"].milestone_network

    # 2. 拓扑指标
    if "isomorphic" in metrics:
        summary["isomorphic"] = calc_isomorphic(net1, net2)
    if "edge_flip" in metrics:
        summary["edge_flip"] = calculate_edge_flip(net1, net2)
    if "him" in metrics:
        summary["him"] = calculate_him(net1, net2)

    # 3. 相关性指标
    if "correlation" in metrics:
        corr_res = calculate_correlation(fadata, ref_model=ref_model, pred_model=now_model)
        # calculate_correlation 返回 dict，取其中的 'correlation' 值
        summary["correlation"] = corr_res.get("correlation", 0.0)

    # 4. 位置预测指标
    pos_keys = ["rf_mse", "rf_rsq", "rf_nmse", "lm_mse", "lm_rsq", "lm_nmse"]
    if any(k in metrics for k in pos_keys):
        pos_res = calculate_position_predict(fadata, ref_model=ref_model, pred_model=now_model, metrics=metrics)
        # 只取用户指定的那些指标
        for k in pos_keys:
            if k in metrics and k in pos_res:
                summary[k] = pos_res[k]

    # 5. 特征重要性指标
    # 5.1 整体相关性
    fi_cor_keys = ["featureimp_cor", "featureimp_wcor"]
    if any(k in metrics for k in fi_cor_keys):
        fic_res = calculate_featureimp_cor(fadata, ref_model=ref_model, pred_model=now_model)
        for k in fi_cor_keys:
            if k in metrics:
                summary[k] = fic_res.get(k, 0.0)

    # 5.2 富集检验
    fi_enr_keys = ["featureimp_ks", "featureimp_wilcox"]
    if any(k in metrics for k in fi_enr_keys):
        fie_res = calculate_featureimp_enrichment(fadata, ref_model=ref_model, pred_model=now_model)
        for k in fi_enr_keys:
            if k in metrics:
                summary[k] = fie_res.get(k, 0.0)

    # 6. 分组映射指标（Cluster metrics）
    if "F1_branch" in metrics:
        cm = calculate_mapping_branches(fadata, simplify=simplify, ref_model=ref_model, pred_model=now_model)
        # calculate_mapping_branches 返回带后缀 "_branches" 的 F1
        summary["F1_branch"] = cm.get("F1_branches", 0.0)
    if "F1_milestone" in metrics:
        cm2 = calculate_mapping_milestones(fadata, simplify=simplify, ref_model=ref_model, pred_model=now_model)
        summary["F1_milestone"] = cm2.get("F1_milestones", 0.0)

    return summary

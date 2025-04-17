from .topology_metric import calc_isomorphic, calculate_edge_flip
from .cluster_metric import calculate_mapping_branches, calculate_mapping_milestones
from cfe.metric._topology_metric.metric_him import calculate_him
from cfe.metric.metric_correlation import calc_correlation
from cfe.metric.metric_position_predict import calculate_position_predict_fadata
from cfe.metric.metric_featureimp import  calculate_overall_feature_importance,calculate_featureimp_cor,calculate_featureimp_enrichment

def calculate_metrics(
        fadata,
        now_model=None,
        ref_model=None,
        simplify=True,
        metrics=["isomorphic", "edge_flip"]
):
    summary_dict = {}
    
    if simplify:
        net1 = fadata.simplify_trajectory(ref_model)["milestone_network"]
        net2 = fadata.simplify_trajectory(now_model)["milestone_network"]
    else:
        net1 = fadata.trajectory_history_dict[ref_model]["milestone_wrapper"]["milestone_network"]
        net2 = fadata.trajectory_history_dict[now_model]["milestone_wrapper"]["milestone_network"]

    #metric correlation
    if "correlation" in metrics:
        summary_dict["correlation"]=calc_correlation()

    # topology metric
    if "isomorphic" in metrics:
        summary_dict["isomorphic"] = calc_isomorphic(net1, net2)
    if "edge_flip" in metrics:
        summary_dict["edge_flip"] = calculate_edge_flip(net1, net2)
    if "him" in metrics:
        summary_dict["him"] = calculate_him(net1, net2)
    
    #position_predict metric
    if any(x in metrics for x in ["rf_mse", "rf_rsq", "rf_nmse", "lm_mse", "lm_rsq", "lm_nmse"]):
        pass

    #featureimp metric
    if any(x in metrics for x in ["featureimp_cor", "featureimp_wcor"]):
        pass

    if any(x in metrics for x in ["featureimp_ks", "featureimp_wilcox"]):
        pass

    # cluster metric
    if "F1_branch" in metrics:
        summary_dict["F1_branch"] = calculate_mapping_branches()
    if "F1_milestone" in metrics:
        summary_dict["F1_milestone"] = calculate_mapping_milestones()

    return summary_dict

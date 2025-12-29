import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import logger


def visualize(benchmark_df, save="benchmark_heatmap.pdf", figsize=(32, 24)):
    import funkyheatmappy

    benchmark_df["id"] = benchmark_df.index

    plt.figure(figsize=figsize)

    column_list = [
        ["id", "group", "name", "geom", "options", "palette"],
        ["id", np.nan, "", "text", {"ha": 0, "width": 4}, np.nan],
        # method meta data
        # ["type", np.nan, "Wrapper Type", "text", {"ha": 0, "width": 4}, np.nan],
        # ["prior_level", np.nan, "Prior Level", "text", {"ha": 0, "width": 4}, np.nan],
    ]

    if "overall" in benchmark_df.columns:
        column_list.append(["overall", "overall", "Overall", "bar", {"width": 4, "legend": False}, "overall_palette"])

    # Accuracy metrics
    id2name = {
        "pseudotime_correlation": "Pseudotime Correlation",
        "velocity_cbdir": "Velocity CBDir",
        "velocity_icvcoh": "Velocity ICVCoh",
        "edge_flip": "Edge Flip",
        "him": "HIM",
        # "isomorphic": "Isomorphic", # 0 for all rows may result in funkyheatmap error
        "F1_branches": "F1 Branches",
        "F1_milestones": "F1 Milestones",
        "correlation": "Correlation",
        "rf_mse": "RF MSE",
        "rf_nmse": "RF NMSE",
        "rf_rsq": "RF R^2",
        "lm_nmse": "LM NMSE",
        "lm_mse": "LM MSE",
        "lm_rsq": "LM R^2",
        "featureimp_cor": "Feature Imp Correlation",
        "featureimp_wcor": "Feature Imp Weighted Correlation",
    }

    metric_column_list = []
    for id, name in id2name.items():
        if id in benchmark_df.columns:
            metric_column_list.append([id, "metric", name, "funkyrect", {"size": 3}, "metric_palette"])
    column_list = column_list + metric_column_list

    # resource group
    if "time" in benchmark_df.columns:
        column_list.append(["time", "resource", "Time", "rect", "scaling", "resource_palette"])
        column_list.append(["time_text", "resource", "", "text", {"overlay": True, "size": 3, "scale": False}, "white6black4"])

    if "memory" in benchmark_df.columns:
        column_list.append(["memory", "resource", "Memory", "rect", "scaling", "resource_palette"])
        column_list.append(["memory_text", "resource", "", "text", {"overlay": True, "size": 3, "scale": False}, "white6black4"])

    column_info = pd.DataFrame(column_list[1:], columns=column_list[0])
    column_info.index = column_info["id"]

    column_groups = pd.DataFrame(
        columns=["Category", "group", "palette"],
        data=[["Overall", "overall", "overall_palette"], ["Metric", "metric", "metric_palette"], ["Resources", "resource", "resource_palette"]],
    )
    # palette definitions
    palettes = {
        "overall_palette": "Blues",
        "metric_palette": "Greens",
        "resource_palette": "YlOrBr",
    }

    funkyheatmappy.funky_heatmap(
        benchmark_df,
        column_info=column_info,
        column_groups=column_groups,
        palettes=palettes,
    )
    if save is not None:
        if isinstance(save, bool) and save:
            save = ".cafe/img/benchmark_funkyheatmap.png"
        plt.savefig(save, bbox_inches="tight")
        logger.debug(f"save trajectory plot to '{save}'")

import os
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from .. import logger
from ..metric import calculate_metrics
from ..metric import metrics as metric_meta_df  # metric dataframe


def metrics(
    fadata,
    model_name_list=None,
    metrics: list = None,
    cluster_edges: list = None,
    metric_dir: str = None,
    overall_score_func: Callable | bool = None,
    if_normalize: bool = True,
    if_impute: bool = True,
    if_save: bool = True,
):
    """ """

    if metrics is None:
        metrics = metric_meta_df["metric_id"].tolist() + ["time", "time_text", "memory", "memory_text"]
    if metric_dir is None:
        metric_dir = f".cafe/{fadata.id}/metric/"

    if cluster_edges is None:
        cluster_edges = fadata.get_milestone_wrapper("ref").milestone_network[["from", "to"]].values.tolist()

    if model_name_list is None:
        method_yaml_file = f".cafe/{fadata.id}/benchmark/methods.yaml"
        with open(method_yaml_file, "r") as f:
            method_parameter_dict = yaml.safe_load(f)
        model_name_list = list(method_parameter_dict.keys())
        logger.info(f"no model_name_list provided, use all methods in method yaml file: {model_name_list}")

    # TODO: metric trajectory dict id check
    # check existing metric files, if missing metrics, recalculate the specific ones
    metric_file_list = os.listdir(metric_dir)
    todo_model_name_list = []
    calculated_metric_df_list = []
    for model_name in model_name_list:
        metric_file = f"{model_name}.csv"
        if metric_file in metric_file_list:
            logger.info(f"metric file for model '{model_name}' already exists. read it.")
            calculated_metric_df = pd.read_csv(f"{metric_dir}/{metric_file}", index_col=0).T
            missing_metrics = [m for m in metrics if m not in calculated_metric_df.columns or pd.isna(calculated_metric_df[m].iloc[0])]
            calculated_metric_df = calculated_metric_df[list(set(metrics) - set(missing_metrics))]
            # check missing metrics
            if missing_metrics:
                logger.info(f"model {model_name} missing metrics: {missing_metrics}, recalculating...")
                fadata.load_trajectory_dict(model_name_list=[model_name])  # load only needed model
                # only calculate missing metrics
                new_metric_df = calculate_metrics(fadata, now_models=[model_name], metrics=missing_metrics, cluster_edges=cluster_edges)
                fadata.remove_trajectory_dict(model_name_list=[model_name])  # remove models after calculation to save memory
                # merge new and old results
                for m in missing_metrics:
                    calculated_metric_df[m] = new_metric_df.loc[model_name, m]
                calculated_metric_df.T.to_csv(f"{metric_dir}/{model_name}.csv")  # save the updated csv
            calculated_metric_df_list.append(calculated_metric_df)
        else:
            todo_model_name_list.append(model_name)

    if calculated_metric_df_list:
        metric_df = pd.concat(calculated_metric_df_list)
    else:
        metric_df = pd.DataFrame()

    # transfer metric columns type to float
    for col in metric_df.columns:
        if col not in ["time_text", "memory_text"]:
            metric_df[col] = metric_df[col].astype(float)

    # calculate new metric
    if "ref" in todo_model_name_list:
        todo_model_name_list.remove("ref")

    if len(todo_model_name_list) > 0:
        logger.info(f"calculating metrics for models: {todo_model_name_list}")
        fadata.load_trajectory_dict(model_name_list=todo_model_name_list)  # load only needed models
        new_metric_df = calculate_metrics(fadata, now_models=todo_model_name_list, metrics=metrics, cluster_edges=cluster_edges)
        fadata.remove_trajectory_dict(model_name_list=todo_model_name_list)  # remove models after calculation to save memory
        new_metric_df.apply(lambda x: x.to_csv(f"{metric_dir}/{x.name}.csv"), axis=1)
        metric_df = pd.concat([metric_df, new_metric_df])

    # normalize the metrics
    if if_normalize:
        metric_df_normalized = pd.DataFrame()
        for metric_name in metric_df.columns:
            if metric_name not in metric_meta_df["metric_id"].to_list():
                if metric_name not in ["time", "memory", "time_text", "memory_text"]:
                    # resource usage metrics are not normalized, but other unknown metrics should warn
                    logger.warning(f"metric '{metric_name}' not found in metric metadata, skipping normalization.")
                metric_df_normalized[metric_name] = metric_df[metric_name]
                continue

            perfect = metric_meta_df.loc[metric_name, "perfect"]
            worst = metric_meta_df.loc[metric_name, "worst"]
            metric_normalized = metric_df[metric_name]
            valid_row = ~metric_normalized.isna()  # valid rows without NaN
            if valid_row.any():
                impute_value = (worst + perfect) / 2
                log_msg = f"impute NaN value('{impute_value}') shown in {valid_row[~valid_row].index.tolist()} for metric '{metric_name}'"
                if if_impute:
                    logger.warning(log_msg)
                    metric_normalized = metric_normalized.fillna(impute_value)  # fillna with mid value
                    valid_row[valid_row.index] = True  # update valid after imputation
                else:
                    logger.warning(f"don't {log_msg}")
            metric_normalized[valid_row] = (metric_normalized[valid_row] - worst) / (perfect - worst)
            metric_df_normalized[metric_name] = metric_normalized
        metric_df = metric_df_normalized

    # order the metric dataframe
    metric_df = metric_df.loc[:, metrics]

    if callable(overall_score_func):
        # custom overall score function
        metric_df["overall"] = metric_df.apply(overall_score_func, axis=1)
    elif overall_score_func is True:
        # default overall score: mean of all metrics
        metric_df["overall"] = metric_df.apply(default_overall_score_func, axis=1)
    else:
        # for False or None
        pass

    if if_save:
        metric_df.to_csv(f"{metric_dir}/summary.csv")  # save summary metric

    return metric_df


# default overall score function, need normalized before
def default_overall_score_func(row):
    target_metrics = [
        "pseudotime_correlation",
        "velocity_cbdir",
        "him",
        "F1_branches",
    ]
    valid_list = []
    for metric in target_metrics:
        if metric not in row:
            logger.warning(f"metric '{metric}' missing in overall score calculation.")
        elif row[metric] == 0 or pd.isnull(row[metric]):
            logger.warning(f"metric '{metric}' is zero or NaN in overall score calculation, overall score set to 0.")
        else:
            valid_list.append(row[metric])

    if valid_list:
        overall = np.prod(valid_list) ** (1 / len(valid_list))
    else:
        overall = 0

    return overall

import os

import yaml

from .. import logger
from ..data import FateAnnData
from ..method import FateMethod
from ..plot import plot_trajectory


def run(
    fadata: FateAnnData,
    method_parameter_dict: dict = None,
    method_yaml_file: str = None,
    load_cached: bool = True,
):
    """
    Run trajectory inference methods in batch according to the parameter dictionary.
    Args:
        fadata: AnnData or compatible object
        method_parameter_dict: dict, method name to parameter dict mapping (optional)
        method_yaml_file: str, yaml file path to load method_parameter_dict if not provided
        load_cached: bool, whether to load cached results if they exist
    """

    # Load method parameter dict from yaml if not provided
    if method_parameter_dict is None:
        if method_yaml_file is None:
            method_yaml_file = f".cafe/{fadata.id}/benchmark/methods.yaml"
            logger.info(f"no method_parameter_dict or method_yaml_file provided, use default method yaml file({method_yaml_file})")
        logger.info(f"load method_parameter_dict from yaml file '{method_yaml_file}'")
        with open(method_yaml_file, "r") as f:
            method_parameter_dict = yaml.safe_load(f)

    # Iterate over each method and its parameters
    for method_name, parameters in method_parameter_dict.items():
        parameters = parameters or {}  # parameters may be None or empty dict

        logger.info(f"running method: {method_name} with params: {parameters}")
        # Check if result already exists
        result_path = f".cafe/{fadata.id}/trajectory_dict/{method_name}.pkl"
        if os.path.exists(result_path):
            if load_cached:
                logger.info(f"result for {method_name} already exists, load cached result")
                continue
            else:
                logger.info(f"result for {method_name} already exists, re-running method")

        # Run the method with given parameters
        method = FateMethod(method_name=method_name)
        method.infer_trajectory(fadata, parameters=parameters, id=method_name)
        plot_trajectory(fadata, save=True)
        fadata.write_trajectory_dict(model_name_list=[method_name])
        logger.info(f"method {method_name} finished and result saved.")

    logger.info("all methods have run.")
    return fadata

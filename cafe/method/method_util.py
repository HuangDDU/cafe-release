import importlib.util
import inspect
import os

import pandas as pd


def load_function(function_name):
    # Get the function from the module
    function_file_path = f"{os.path.dirname(__file__)}/function/cf_{function_name}.py"
    spec = importlib.util.spec_from_file_location(function_name, function_file_path)  # Load the module
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function_obj = getattr(module, function_name)
    return function_obj


def get_function_parameter_dict(function_obj):
    # extract defined paramters from function.
    sig = inspect.signature(function_obj)
    parameter_dict = {}
    for param_name, param in sig.parameters.items():
        param_info = {
            "name": param_name,
            "required": param.default is inspect.Parameter.empty,
            "default": param.default if param.default != inspect.Parameter.empty else None,
            "annotation": param.annotation if param.annotation != inspect.Parameter.empty else None,
            "kind": param.kind.name,
        }
        parameter_dict[param_name] = param_info
    return parameter_dict


def get_function_method_info(function_obj):
    method_info = getattr(function_obj, "_method_info", {})
    return method_info


def scan_method(keep_available_only=True):
    methods_dict = {}
    function_dir = f"{os.path.dirname(__file__)}/function/"  # scan th dir

    for py_file in os.listdir(function_dir):
        if not (py_file.startswith("cf_") and py_file.endswith(".py")):
            # only import cf_xxx.py file, skip others
            continue

        function_name = py_file[3:-3]  # remove 'cf_' prefix and '.py' suffix

        function_obj = load_function(function_name)
        parameter_dict = get_function_parameter_dict(function_obj)
        method_info = get_function_method_info(function_obj)
        method_info["parameter"] = parameter_dict

        methods_dict[function_name] = method_info

        # TODO: Github star and google scholar citations statistics
    method_df = pd.DataFrame(methods_dict).T
    method_df["available"] = method_df["available"].fillna("False")
    if keep_available_only:
        method_df = method_df.query("available==True")

    return method_df

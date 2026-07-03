import importlib.util
import inspect
import os
import subprocess

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


def _get_local_docker_images() -> set[str]:
    """Return local docker images as repository:tag strings."""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}
    except Exception:
        return set()


def _build_dynverse_parameter_dict(inputs_df: pd.DataFrame) -> dict:
    """Convert dynverse definition inputs to cafe-style parameter metadata."""
    prior_map = {
        "groups_id": "cluster",
        "start_id": "start_cell",
        "dimred": "basis",
    }

    parameter_dict = {}
    for _, row in inputs_df.iterrows():
        input_id = str(row.get("input_id", ""))
        input_type = str(row.get("type", ""))
        required = bool(row.get("required", False))

        if input_type not in {"prior_information", "parameter"}:
            continue

        key = prior_map.get(input_id, input_id)
        parameter_dict[key] = {
            "name": key,
            "required": required,
            "default": None,
            "annotation": None,
            "kind": "KEYWORD_ONLY",
        }

    return parameter_dict


def _scan_dynverse_method(method_backend_df: pd.DataFrame, keep_available_only: bool) -> pd.DataFrame:
    """Scan dynverse methods declared in method_backend.csv."""
    dynverse_rows = method_backend_df[method_backend_df["dynverse_docker"].notna()].copy()
    dynverse_rows = dynverse_rows[dynverse_rows["dynverse_docker"].astype(str).str.strip() != ""]
    if dynverse_rows.empty:
        return pd.DataFrame()

    local_images = _get_local_docker_images()
    methods_dict = {}

    for method_name, row in dynverse_rows.iterrows():
        image_id = str(row["dynverse_docker"]).strip()
        available = image_id in local_images if local_images else False

        if keep_available_only and (not available):
            continue

        method_info = {
            "name": method_name,
            "version": image_id.split(":")[-1] if ":" in image_id else None,
            "description": f"Dynverse docker method: {method_name}",
            "wrapper_type": None,
            "doi": None,
            "github_url": None,
            "use_gpu": False,
            "cpu_parallelization": True,
            "available": available,
            "parameter": {},
            "backend": "dynverse_docker",
            "docker_image": image_id,
        }

        if available:
            try:
                from .fate_method import FateMethod

                method = FateMethod(method_name=method_name, backend_name="dynverse_docker")
                method.choose_backend("dynverse_docker")
                definition = method.method_backend.definition

                if "method" in definition and isinstance(definition["method"], dict):
                    method_info["name"] = definition["method"].get("name", method_info["name"])

                if "wrapper" in definition and isinstance(definition["wrapper"], dict):
                    wrapper_type = definition["wrapper"].get("type", None)
                    if isinstance(wrapper_type, list):
                        method_info["wrapper_type"] = wrapper_type[0] if len(wrapper_type) > 0 else None
                    else:
                        method_info["wrapper_type"] = wrapper_type

                if "manuscript" in definition and isinstance(definition["manuscript"], dict):
                    method_info["doi"] = definition["manuscript"].get("doi", None)

                inputs_df = definition.get_inputs_df()
                method_info["parameter"] = _build_dynverse_parameter_dict(inputs_df)
            except Exception:
                # Keep a conservative fallback row if definition cannot be loaded.
                pass

        methods_dict[method_name] = method_info

    if not methods_dict:
        return pd.DataFrame()
    return pd.DataFrame(methods_dict).T


def scan_method(keep_available_only=True, scan_dynverse_method=False):
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

    if scan_dynverse_method:
        method_backend_filename = f"{os.path.dirname(__file__)}/method_backend.csv"
        method_backend_df = pd.read_csv(method_backend_filename, index_col=0)
        dynverse_method_df = _scan_dynverse_method(method_backend_df, keep_available_only=keep_available_only)
        if not dynverse_method_df.empty:
            all_cols = sorted(set(method_df.columns).union(set(dynverse_method_df.columns)))
            method_df = method_df.reindex(columns=all_cols)
            dynverse_method_df = dynverse_method_df.reindex(columns=all_cols)
            method_df = pd.concat([method_df, dynverse_method_df], axis=0)

    return method_df

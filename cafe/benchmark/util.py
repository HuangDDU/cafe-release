import os
import subprocess

import pandas as pd
import yaml

from .._logging import logger
from ..method.method_util import scan_method


def generate_method_paramters(
    fadata,
    filter_device: str = "all",  # gpu, cpu, all
    check_backend: bool = True,
):
    # generate parameter dict yaml file in the benchmark result  directory(.cafe/pancreas/benchmark/)

    method_backend_df = pd.read_csv(f"{os.path.dirname(__file__)}/../method/method_backend.csv", index_col=0)  # method backend dataframe
    method_meta_df = scan_method()  # method meta dataframe, "use_gpu", "available" column included
    valid_method_name = [
        x for x in method_backend_df.index if x in method_meta_df.index or "ti_" in x
    ]  # keep available methods, or methods with dynverse docker image (ti_xxx)
    method_df = pd.DataFrame(index=valid_method_name, columns=["use_gpu", "backend_name"])
    method_df = method_df.loc[valid_method_name]
    method_df.loc[method_meta_df.index, "use_gpu"] = method_meta_df["use_gpu"]
    method_df["use_gpu"] = method_df["use_gpu"].fillna(False)
    # remove unavailable methods, only keep methods with available backend meta and dynverse docker image

    if filter_device == "gpu":
        method_df = method_df.query("use_gpu == True")
    elif filter_device == "cpu":
        method_df = method_df.query("use_gpu == False")
    elif filter_device == "all":
        logger.debug("select all methods to generate")
    else:
        logger.warning(f"filter_device: {filter_device} is not available, using default all")

    if check_backend:
        conda_env_list = get_available_conda_env()
        docker_image_list = get_available_docker_backend()
        # docker_image_list = [docker_image.split("/")[-1].split(":")[0] for docker_image in docker_image_list if "dynverse" in docker_image] # only for dynverse docker

        def _check_backend(row):
            # Prioritize conda, then docker. Adjust logic if needed.
            # Assuming 'conda' column contains the env name and 'dynverse_docker' contains image tag
            conda_env = row["conda"]
            docker_img = row["dynverse_docker"]

            if pd.notna(conda_env) and conda_env in conda_env_list:
                return "conda"
            elif pd.notna(docker_img) and docker_img in docker_image_list:
                return "dynverse_docker"
            else:
                return None

        method_df["backend_name"] = method_backend_df.apply(_check_backend, axis=1)

        # Log warnings for missing backends
        missing_backend_methods = method_df[method_df["backend_name"].isna()].index.tolist()
        if missing_backend_methods:
            logger.warning(
                f"The following methods were skipped due to missing backends (Conda env or Docker image): {', '.join(missing_backend_methods)}"
            )

        # Filter to keep only valid methods
        method_df = method_df.dropna(subset=["backend_name"])

    parameter_dict = {method_name: "" for method_name in method_df.index.tolist()}
    # Ensure benchmark dir exists
    os.makedirs(f"{fadata.benchmark_dir}", exist_ok=True)
    with open(f"{fadata.benchmark_dir}/methods.yaml", "w") as f:
        yaml.dump(parameter_dict, f)
    return method_df


def get_available_conda_env():
    """Get list of available conda environments."""
    try:
        # Use conda env list --json for easier parsing, or fallback to text parsing
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
        envs = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Logic to extract env name (first column)
            parts = line.split()
            if parts:
                envs.append(parts[0])
        return envs
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to list conda environments. Is conda installed and in PATH?")
        return []


def get_available_docker_backend():
    """Get list of available docker images."""
    try:
        # Format: repository:tag
        result = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True)
        if result.returncode != 0:
            # Docker might not be running or permissions issue
            logger.warning(f"Docker check failed: {result.stderr}")
            return []

        images = [line.strip() for line in result.stdout.splitlines() if line.strip()]

        # Also add image IDs or repository names without tags if dynverse uses strict naming
        # Depending on how 'ti_xxx' is stored in csv, you might need just the repo name
        result_repo = subprocess.run(["docker", "images", "--format", "{{.Repository}}"], capture_output=True, text=True)
        if result_repo.returncode == 0:
            repos = [line.strip() for line in result_repo.stdout.splitlines() if line.strip()]
            images.extend(repos)

        return list(set(images))
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to list docker images. Is docker installed and running?")
        return []

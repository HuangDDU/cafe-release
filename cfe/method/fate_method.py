import os.path
from typing import Literal, Optional

import pandas as pd

from .._logging import logger, set_log_file
from .._settings import settings
from ..data.fate_anndata import FateAnnData
from ..util import random_time_string
from .fate_cfe_docker_backend import CFEDockerBackend
from .fate_conda_backend import CondaBackend
from .fate_dynverse_docker_backend import DynverseDockerBackend
from .fate_function_backend import FunctionBackend

# import yaml


class FateMethod:
    """FateMethod, available backend: python_function, cfe_docker, dynverse_docker"""

    def __init__(
        self,
        method_name: str = "paga",
        backend_name: Optional[Literal["conda", "python_function", "cfe_docker", "dynverse_docker", None]] = None,
    ):
        """Initialize the FateMethod class.

        Args:
            method_name (str, optional): trajectory inference method name.
            backend (_type_, optional): python_function, cfe_docker, dynverse_docker.
        """
        # logger.debug("FateMethod __init__")
        self.method_name = method_name
        # TODO: backend lazy load: choose backend when the backend is firstly called.
        self.backend_name = backend_name
        self.backend = None

    def choose_backend(
        self,
        backend: Optional[Literal["conda", "python_function", "cfe_docker", "dynverse_docker", None]] = None,
    ) -> None:
        """choose backend according to input backend and method_name
        Args:
            backend (_type_, optional): python_function, cfe_docker, dynverse_docker.
        """
        # logger.debug("FateMethod choose_backend")
        backend = settings.backend if backend is None else backend
        if backend is None:
            # backend in function parameteres and setting file are both None, choose backend
            # input msg for choosing backend
            answer = input(
                """
                    You can run this method as an Python function (1), CFE Docker containter(2), Dynverse Docker container(3, default)
                    Which do you want to use?
                    1: Python function
                    2: CFE Docker Container
                    else: Dynverse Docker Container[default]
            """
            )

            if answer == "1":
                backend = "python_function"
            elif answer == "2":
                backend = "cfe_docker"
            else:
                backend = "dynverse_docker"
            settings["backend"] = backend  # update default backend in setting

        # with open(os.path.join(os.path.dirname(__file__), "method_backend.yml"), "r") as file:
        #     method_backend_dict = yaml.safe_load(file)
        # replace yml by csv, more clearer
        method_backend_dict = pd.read_csv(os.path.join(os.path.dirname(__file__), "method_backend.csv"), index_col=0).T.to_dict()

        # need adjust for some incomplete methods like slingshot, only dynverse docker is available
        if pd.isna(method_backend_dict[self.method_name][backend]):
            # choose backend that have value
            available_backend_list = []
            for k, v in method_backend_dict[self.method_name].items():
                if not pd.isna(v):
                    available_backend_list.append(k)
            new_backend = available_backend_list[-1]
            logger.info(f"backend:'{backend}' is not available for method:'{self.method_name}', choosing new backend: '{new_backend}'")
            backend = new_backend

        # TODO: remove yaml definition file to decrease parameter redundancy , function parameters is sufficient
        if backend == "python_function":
            function_name = method_backend_dict[self.method_name]["python_function"]
            self.method_backend = FunctionBackend(function_name)
        elif backend == "conda":
            function_name = method_backend_dict[self.method_name]["python_function"]
            conda_name = method_backend_dict[self.method_name]["conda"]
            self.method_backend = CondaBackend(function_name, conda_name)
        elif backend == "cfe_docker":
            function_name = method_backend_dict[self.method_name]["python_function"]
            image_id = method_backend_dict[self.method_name]["cfe_docker"]
            self.method_backend = CFEDockerBackend(function_name, image_id)
        elif backend == "dynverse_docker":
            # backend == "dynverse_docker"
            image_id = method_backend_dict[self.method_name]["dynverse_docker"]
            self.method_backend = DynverseDockerBackend(image_id)
        else:
            raise ValueError(f"backend {backend} not supported")
        logger.info(f"method backend loaded: {self.method_backend}")

        self.backend = backend
        # TODO: method info parsed from @method_info

    def infer_trajectory(
        self,
        fadata: FateAnnData,
        parameters: dict = {},
        rewrite: bool = True,
        backend_name: Optional[Literal["conda", "python_function", "cfe_docker", "dynverse_docker", None]] = None,
    ) -> None:
        """call the run function of method backend,

        ref: pydynverse/wrap/method_execute._method_execute

        Args:
            fadata (FateAnnData): dataset.
            parameters (dict, optional): parametre dict. Defaults to {}.

        """
        if (self.backend is None) or ((backend_name is not None) and (backend_name == self.backend_name)):
            backend_name = backend_name if backend_name is not None else self.backend_name  # newer backend
            self.choose_backend(self.backend_name)
            self.id = random_time_string(f"{self.method_name}-{self.backend}")
        if settings.seperate_log_file:
            set_log_file(f".cfe/{fadata.id}-{self.id}.log")
        if rewrite:
            fadata.add_model_name(self.id)

        self.method_backend.run(fadata, parameters)

        if settings.seperate_log_file:
            set_log_file()  # reset to default log file
        logger.info(f"method infer trajectory successfully: {self.method_backend}")
        logger.debug(f"milestone_network: \n {fadata.get_milestone_wrapper().milestone_network}")

    def __call__(self, fadata, parameters, **kwargs):
        self.infer_trajectory(fadata, parameters, **kwargs)

    # TOOD: consider if __call__ is needed
    # def __call__(
    #     self,
    #     fadata: FateAnnData = None,
    #     rewrite: bool = True,
    #     id: str = None,
    #     backend_name: Optional[Literal["conda", "python_function", "cfe_docker", "dynverse_docker", None]] = None,
    #     **parameters,
    # ):
    #     """simplified version for self.infer_trajectory"""
    #     if (self.backend is None) or ((backend_name is not None) and (backend_name != self.backend_name)):
    #         # choose backend firstly or rechoose new backend
    #         backend_name = backend_name if backend_name is not None else self.backend_name  # newer backend
    #         self.choose_backend(backend_name)
    #     if id is None:
    #         self.id = random_time_string(f"{self.method_name}-{self.backend}")
    #     else:
    #         self.id = id
    #     if rewrite:
    #         # use new model name
    #         fadata.add_model_name(self.id)
    #     adata = fadata.to_anndata(delete_trajectory=True)

    #     trajectory_dict = self.method_backend(adata, **parameters)

    #     if "wrapper_type" not in trajectory_dict:
    #         #  if the method have only one wrapper , read from definition yaml file
    #         wrapper_type = self.method_backend.definition["wrapper"]["type"]
    #         trajectory_dict["wrapper_type"] = wrapper_type[0] if isinstance(wrapper_type, list) else wrapper_type

    #     fadata.add_trajectory_by_type(trajectory_dict)

    #     # add resource usage if benchmark_resource is True
    #     if "resource_usage" in trajectory_dict:
    #         fadata.add_resource_usage(trajectory_dict["resource_usage"])

    def __str__(self):
        return f"FateMethod: method_backend-{self.method_backend}, backend-{self.backend}"

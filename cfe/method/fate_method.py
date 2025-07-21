import os.path
from typing import Literal, Optional

# import pandas as pd
import yaml

from .._logging import logger
from .._settings import settings
from ..data.fate_anndata import FateAnnData
from ..util import random_time_string
from .fate_cfe_docker_backend import CFEDockerBackend
from .fate_conda_backend import CondaBackend
from .fate_dynverse_docker_backend import DynverseDockerBackend
from .fate_function_backend import FunctionBackend


class FateMethod:
    """FateMethod, available backend: python_function, cfe_docker, dynverse_docker"""

    def __init__(
        self,
        method_name: str = "paga",
        backend: Optional[Literal["python_function", "cfe_docker", "dynverse_docker", None]] = None,
    ):
        """Initialize the FateMethod class.

        Args:
            method_name (str, optional): trajectory inference method name.
            backend (_type_, optional): python_function, cfe_docker, dynverse_docker.
        """
        # logger.debug("FateMethod __init__")
        self.method_name = method_name
        self.choose_backend(backend)
        self.id = random_time_string(f"{method_name}-{self.backend}")

    def choose_backend(self, backend: Optional[Literal["python_function", "cfe_docker", "dynverse_docker", None]] = None) -> None:
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

        with open(os.path.join(os.path.dirname(__file__), "method_backend.yml"), "r") as file:
            method_backend_dict = yaml.safe_load(file)

        # TODO: need adjust for some incomplete methods like slingshot, only dynverse docker is available
        if backend == "python_function":
            function_name = method_backend_dict[self.method_name]["python_function"]
            self.method_backend = FunctionBackend(function_name)
        elif backend == "conda":
            function_name = method_backend_dict[self.method_name]["python_function"]
            conda_name = method_backend_dict[self.method_name]["conda_env"]
            self.method_backend = CondaBackend(function_name, conda_name)
        elif backend == "cfe_docker":
            image_id = method_backend_dict[self.method_name]["cfe_docker"]
            self.method_backend = CFEDockerBackend(image_id)
        elif backend == "dynverse_docker":
            # backend == "dynverse_docker"
            image_id = method_backend_dict[self.method_name]["dynverse_docker"]
            self.method_backend = DynverseDockerBackend(image_id)
        else:
            raise ValueError(f"backend {backend} not supported")
        logger.info(f"method_backend: {self.method_backend}")

        self.backend = backend

    def infer_trajectory(
        self,
        fadata: FateAnnData = None,
        parameters: dict = None,
    ) -> None:
        """call the run function of method backend,

        ref: pydynverse/wrap/method_execute._method_execute

        Args:
            fadata (FateAnnData, optional): _description_. Defaults to None.
            parameters (dict, optional): _description_. Defaults to None.
        """
        # logger.debug("FateMethod infer_trajectory")
        fadata.add_model_name(self.id)
        self.method_backend.run(fadata, parameters)

    def get_parameter_df(self):
        # show parameters from backend's definition object
        definition = self.method_backend.definition
        return definition.parameters

    # TODO: if prior information is needed?
    # def get_prior_information(self):
    #     # show prior information from backend's definition object
    #     definition = self.method_backend.definition
    #     return definition.wrapper["prior_information"]

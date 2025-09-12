import importlib.util
import inspect
import os
from abc import ABC, abstractmethod

import docker
import tqdm

from .._logging import logger


# Backend: abstract class, used for subsequent specific implementation such as "DockerBackend" class and "FunctionBackend" class
class Backend(ABC):
    @abstractmethod
    def load_backend(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def _load_function(self, function_name):
        # load the method function and extract prameters, will be used in three backend: python_function, conda, cfe_docker

        function_file_path = f"{os.path.dirname(__file__)}/function/{function_name}.py"
        spec = importlib.util.spec_from_file_location(self.function_name, function_file_path)  # Load the module
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function from the module
        function_obj = getattr(module, function_name)
        logger.info(f"Loaded function: {function_obj} from {function_file_path}")
        self.function = function_obj

        # extract defined paramters from function.
        sig = inspect.signature(function_obj)
        function_parameter_dict = {}
        for param_name, param in sig.parameters.items():
            param_info = {
                "name": param_name,
                "default": param.default if param.default != inspect.Parameter.empty else None,
                "annotation": param.annotation if param.annotation != inspect.Parameter.empty else None,
                "kind": param.kind.name,
            }
            function_parameter_dict[param_name] = param_info
        self.function_parameter_dict = function_parameter_dict

    def _get_parameters(self, fadata, parameters):
        # merge parameters and extracted prior information as completed prameters. it should be called after "_load_function"

        # extract valid prior information, remove undefined parameter
        prior_information_valid = set(fadata.prior_information.keys()) & set(self.function_parameter_dict.keys())
        if prior_information_valid:
            logger.debug(f"extract prior information automatically: {prior_information_valid}")
        parameters_undefined = set(parameters.keys()) - set(self.function_parameter_dict.keys())
        if parameters_undefined:
            logger.info(f"remove undefined parameters: {parameters_undefined}")
            for parameter_name in parameters_undefined:
                del parameters[parameter_name]

        # merge them, manually specified parameters have higher priority than automatically extracted prior knowledge
        for k in prior_information_valid:
            if k not in parameters:
                parameters[k] = fadata.prior_information[k]
        logger.debug(f"merged parameters: {parameters}")
        return parameters


class DockerBackend(Backend):
    def _load_image(self):
        """
        ref: pydynverse.wrap.method_create_ti_method_container.create_ti_method_container
        """
        image_id = self.image_id
        # load dynverse docker image
        client = docker.from_env()

        # check docker image exists
        try:
            # exist
            img = client.images.get(image_id)
            self.entrypoint = img.attrs["Config"]["Entrypoint"]
            logger.debug(f"Docker image({image_id}) loaded, entry point is '{self.entrypoint}'")
        except Exception as e:
            # no exist, need pull request
            logger.debug(e)
            logger.info(f"Docker image({image_id}) was not found")
            # client.images.pull(container_id)
            image_name, tag = image_id.split(":")
            self._pull_image_with_progress(image_name, tag=tag, logger_func=logger.info)
            img = client.images.get(image_id)
            logger.info(f"Docker image({image_id}) {img} loaded")

    def _pull_image_with_progress(self, image_name, tag=None, logger_func=print):
        """
        pull dynverse docker image  and show progress bar with tqdm

        ref: pydynverse.wrap.method_create_ti_method_container.pull_image_with_progress
        """
        if logger_func is None:
            # default logger function is print
            logger_func = print
        client = docker.from_env()
        try:
            logger_func(f"Try to pull image {image_name}:{tag}...\n")
            api_client = docker.APIClient(base_url="unix://var/run/docker.sock")  # stream docker clinet can get log
            pull_logs = api_client.pull(repository=image_name, tag=tag, stream=True, decode=True)  # pull image
            progress_bars = {}  # initialize progress bar, store every layer's progress bar
            for log in pull_logs:
                # pull logs format is JSON, need parse
                if "status" in log:
                    status = log["status"]
                    layer_id = log.get("id", None)
                    progress_detail = log.get("progressDetail", {})
                    current = progress_detail.get("current", 0)  # finished bytes
                    total = progress_detail.get("total", 0)  # total bytes
                    # layer_id and total update progress bar
                    if layer_id and total:
                        if layer_id not in progress_bars:
                            # new propgress bar
                            progress_bars[layer_id] = tqdm(total=total, desc=f"Layer {layer_id[:12]}", unit="B", unit_scale=True, unit_divisor=1024)
                        progress_bars[layer_id].n = current
                        progress_bars[layer_id].refresh()
                    # no progression information, show status
                    elif layer_id:
                        logger_func(f"{status} {layer_id}".strip())
                    else:
                        logger_func(f"{status}".strip())
            # close all progress bars
            for bar in progress_bars.values():
                bar.close()
            logger_func(f"Pull {image_name}:{tag} finish")
        except docker.errors.APIError as e:
            logger_func(f"Pull image failed: {e}")
        except Exception as e:
            logger_func(f"Other Error: {e}")
        finally:
            client.close()

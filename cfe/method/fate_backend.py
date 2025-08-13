import tempfile
from abc import ABC, abstractmethod

import docker
import pandas as pd
import tqdm
import yaml

from .._logging import logger


# Backend: abstract class, used for subsequent specific implementation such as "DockerBackend" class and "FunctionBackend" class
class Backend(ABC):
    @abstractmethod
    def load_backend(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _load_definition(self):
        """_summary_"""
        pass

    # Note: Only used for dynverse docker backend"
    def _extract_prior_information(self, fdata, inputs_df):
        """
        ref: PyDynverse/pydynverse/wrap/method_extract_args.py _method_extract_priors
        """
        # logger.debug("FateMethod _extract_prior_information")

        # extract prior information from
        priors = fdata.prior_information
        priors_key_list = priors.keys()
        priors_key_set = set(priors_key_list)

        # check required priors
        required_prior_ids = inputs_df["input_id"][inputs_df["required"] & (inputs_df["type"] == "prior_information")].tolist()
        required_prior_ids_set = set(required_prior_ids)
        if not (required_prior_ids_set <= priors_key_set):
            # all required priors are needed, if not , raise error
            missing_priors = required_prior_ids_set - priors_key_set
            msg = f"""
                ! Prior information {','.join(missing_priors)} is missing from dataset {fdata.id} but is required by the method. \n
                -> If known, you can add this prior information using fadata.add_prior_information({' ,'.join([str(i)+' = <prior>' for i in missing_priors])}). \n
                -> Otherwise, this method cannot be used.
            """
            raise Exception(msg)
        required_prior = {k: priors[k] for k in required_prior_ids}

        # check optional priors
        optional_prior_ids = inputs_df["input_id"][(~inputs_df["required"]) & (inputs_df["type"] == "prior_information")].tolist()
        optional_prior_ids_set = set(optional_prior_ids)
        if not (optional_prior_ids_set <= priors_key_set):
            # all required priors are not needed, enven if not all are provided, only warning
            missing_priors = list(optional_prior_ids_set - priors_key_set)
            msg = f"""
                Prior information {','.join(missing_priors)} is optional, but missing from dataset {fdata.id}. \n
                Will not give this prior to method.
            """
            logger.warning(msg)
        optional_prior = {k: priors[k] for k in list(optional_prior_ids_set & priors_key_set)}  # remove irrelevant keys

        priors = required_prior | optional_prior

        return priors


class DockerBackend(Backend):
    def load_backend(self):
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
            logger.debug(f"Docker image({image_id}) loaded")
        except Exception as e:
            # no exist, need pull request
            logger.debug(e)
            logger.info(f"Docker image({image_id}) was not found")
            # client.images.pull(container_id)
            image_name, tag = image_id.split(":")
            self._pull_image_with_progress(image_name, tag=tag, logger_func=logger.info)
            img = client.images.get(image_id)
            logger.info(f"Docker image({image_id}) {img} loaded")

        self._load_definition()  # load definition

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

    def _load_definition(self):
        """
        extract and parse definition.yml, including description, required parameters and prior knowledge

        ref: pydynverse.wrap.container_get._container_get_definition
        """
        with tempfile.TemporaryDirectory() as tmp_wd:
            # start docker container
            client = docker.from_env()
            container = client.containers.run(
                entrypoint="cp /code/definition.yml /copy_mount/definition.yml",  # aim copy dir
                image=self.image_id,
                volumes=[f"{tmp_wd}:/copy_mount"],
                detach=True,
            )
            container.wait()
            container.stop()
            container.remove()
            # read and parse yml file
            with open(f"{tmp_wd}/definition.yml", "r") as file:
                definition_raw = yaml.safe_load(file)

        definition = Definition(definition_raw)
        definition["run"] = {"backend": "container", "image_id": self.image_id}
        self.definition = definition


class Definition:
    def __init__(self, definition_raw: dict):
        self.method = definition_raw["method"]
        self.wrapper = definition_raw["wrapper"]
        self.container = definition_raw["container"]
        self.package = definition_raw["package"] if "package" in definition_raw else None
        self.manuscript = definition_raw["manuscript"] if "manuscript" in definition_raw else None
        self.parameters = pd.DataFrame(definition_raw["parameters"]).set_index("id")
        self.run = {}

        # inputs
        inputs = self.wrapper["input_required"]
        inputs = inputs.copy() if isinstance(inputs, list) else [inputs]
        if "input_optional" in self.wrapper:
            input_optional = self.wrapper["input_optional"]
            inputs += input_optional if isinstance(input_optional, list) else [input_optional]

        # extra input, including data and parameters
        params = self.parameters.index.tolist()
        input_id_list = inputs + params
        required_list = [i in self.wrapper["input_required"] for i in input_id_list]
        type_list = []
        for input_id in input_id_list:
            # type column
            if input_id in ["counts", "expression", "expression_future"]:
                type_list.append("expression")
            elif input_id in params:
                type_list.append("parameter")
            else:
                type_list.append("prior_information")
        inputs_df = pd.DataFrame({"input_id": input_id_list, "required": required_list, "type": type_list})
        self.wrapper["inputs"] = inputs_df

    def get_inputs_df(self):
        return self.wrapper["inputs"]

    def get_parameters(self, new_parameters=None):
        default_parameters = self.parameters["default"].to_dict()
        if new_parameters is None:
            # return default parameters
            return default_parameters
        else:
            # merge new parameters and default parameters to get parameters
            parameters = default_parameters
            # parameters.update(new_parameters)
            #
            for k, v in new_parameters.items():
                if k in parameters:
                    if isinstance(v, dict) and self.parameters.loc[k, "update"]:
                        #  for dict parameters, it should be updated by merge but not to replace.
                        parameters[k].update(v)
                    else:
                        parameters[k] = v
                else:
                    logger.warning(f"{k} is not a valid parameter")
            return parameters

    def add_function_wrapper(self, return_function):
        if not return_function:
            # 直接返回字典格式
            return self
        # else:
        #     # 返回函数格式，等待默认参数进一步设置
        #     defaults = get_default_parameters(definition)  # 获取代码函数中的参数默认值

        #     def param_overrider_fun(**kwargs):
        #         # 参数覆盖, 接受代码函数中的参数默认值传入参数并覆盖definition.yml
        #         new_defaults = kwargs
        #         param_names = list(definition["parameters"].index)
        #         for param_name, v in new_defaults.items():
        #             if param_name in param_names:
        #                 definition["parameters"].loc[param_name, "default"] = v
        #             else:
        #                 # 该参数不definition.yml文件里定义
        #                 logger.error(f"Unknown parameter: {param_name}")
        #         return definition

        #     param_overrider_fun.__kwdefaults__ = defaults

        #     return param_overrider_fun

    def __contains__(self, item):
        "check if have attribute"
        return hasattr(self, item)

    def keys(self):
        """return all attibute name, then the function dict() can be used"""
        return self.__dict__.keys()

    def __getitem__(self, key):
        "get attribute"
        return getattr(self, key)

    def __setitem__(self, key, value):
        "set attribute"
        setattr(self, key, value)

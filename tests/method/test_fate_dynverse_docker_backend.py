import os.path

import docker
import pandas as pd
import pytest
import scipy.sparse as sp

import cfe

from .test_fate_function_backend import get_test_run_data

image_id = "dynverse/ti_comp1:v0.9.9.01"


@pytest.mark.local
class TestDynverseDockerBackend:
    def setup_method(self):
        self.dynverse_docker = cfe.method.DynverseDockerBackend(image_id)

    def test_init(self):
        assert self.dynverse_docker.image_id == image_id

    def test_load_backend(self):
        # load_backend has benn called in __init__, implemented in DockerBackend
        assert self.dynverse_docker.definition is not None

    def test_run(self):
        fadata, parameters = get_test_run_data()
        self.dynverse_docker.run(fadata, parameters)
        assert fadata.is_wrapped_with_trajectory

    def test_pull_image_with_progress(self):
        # _pull_image_with_progress is called in test_load_backend, which is called in __init__
        # check if the specific image has been downloaded
        client = docker.from_env()
        flag = False
        for image in client.images.list():
            if image_id in image.tags:
                flag = True
        assert flag

    def test_load_definition(self):
        # _load_definition is called in test_load_backend, which is called in __init__
        from cfe.method.fate_dynverse_docker_backend import Definition

        definition = self.dynverse_docker.definition
        assert isinstance(definition, Definition)


# ====================================================================================================
# YAML ubject for 'definition.yaml' file


@pytest.mark.local
class TestDefinition:
    #
    def setup_method(self):
        # definition in dynverse docker backend

        # static definition file
        # from cfe.method.fate_dynverse_docker_backend import Definition
        # definition_yaml_filename = f"{os.path.dirname(__file__)}/../../cfe/method/definition_deprecated/cf_paga.yml"
        # with open(definition_yaml_filename, "r") as file:
        #     definition_raw = yaml.safe_load(file)
        # self.definition = Definition(definition_raw)

        # dynamic definition file need to be download from docker container
        dynverse_docker = cfe.method.DynverseDockerBackend(image_id)
        self.definition = dynverse_docker.definition

    def test_magic_methods(self):
        definition = self.definition

        definition["run"]

        # test __contains__
        assert "method" in definition, "method should in definition"
        # test __getitem__
        assert definition["method"] == definition.method, "definition['method'] should be the same as definition.method"
        # test keys
        definition_dict = dict(definition)
        attribute_name_list = ["method", "wrapper", "container", "package", "manuscript", "parameters"]
        assert set(attribute_name_list).issubset(
            set(definition_dict.keys())
        ), f"{attribute_name_list} should be the keys of the dict: {definition_dict}"

    def test_get_inputs_df(self):
        inputs_df = self.definition.get_inputs_df()
        assert isinstance(inputs_df, pd.DataFrame)
        assert inputs_df.columns.tolist() == ["input_id", "required", "type"]
        assert inputs_df["type"].isin(["expression", "parameter", "prior_information"]).all()

    def test_get_parameters(self):
        parameters = self.definition.get_parameters()
        assert isinstance(parameters, dict)


# ====================================================================================================
# Dynverse Docker I/O Object
data_dir = f"{os.path.dirname(__file__)}/middle_file"

# middle file
input_json_filename = f"{data_dir}/input.json"
input_h5_filename = f"{data_dir}/input.h5"
output_h5_filename = f"{data_dir}/output.h5"
output_json_filename = f"{data_dir}/output.json"

# data
inputs = {
    "expression_id": "count",
    "expression": sp.csr_matrix([[1, 0], [0, 1]]),
    "cell_ids": ["cell1", "cell2"],
    "feature_ids": ["gene1", "gene2"],
    "parameters": {"param1": 0.1},
    "priors": {"start_cell": "cell1"},
    "seed": 0,
    "verbose": True,
}


@pytest.mark.local
class TestDynverseDockerInput:
    def setup_method(self):
        from cfe.method.fate_dynverse_docker_backend import DynverseDockerInput

        dynverse_docker_input_obj = DynverseDockerInput(**inputs)

        dynverse_docker_input_obj.input_json_filename = input_json_filename
        dynverse_docker_input_obj.input_h5_filename = input_h5_filename
        self.dynverse_docker_input_obj = dynverse_docker_input_obj

    def test_save_json(self):
        self.dynverse_docker_input_obj.save_json(input_json_filename)

    def test_json2h5(self):
        self.dynverse_docker_input_obj.json2h5(input_h5_filename)

    def test_write_h5_directly(self):
        self.dynverse_docker_input_obj.write_h5_directly(input_h5_filename)


@pytest.mark.local
class TestDynverseDockerOutput:
    def setup_method(self):
        from cfe.method.fate_dynverse_docker_backend import DynverseDockerOutput

        self.dynverse_docker_output_obj = DynverseDockerOutput()
        self.output_h5_filename = output_h5_filename
        self.output_json_filename = output_json_filename

    def test_h52json(self):
        self.dynverse_docker_output_obj.h52json(output_h5_filename, output_json_filename)

    def test_load_json(self):
        self.dynverse_docker_output_obj.h52json(output_h5_filename, output_json_filename)

    def test_load_h5_directly(self):
        self.dynverse_docker_output_obj.load_h5_directly(output_h5_filename)


@pytest.mark.local
def test_write_h5():
    from cfe.method.fate_dynverse_docker_backend import write_h5

    # minor fix for DynverseDockerInput __init__
    new_input = inputs.copy()
    new_input[new_input["expression_id"]] = new_input["expression"]
    del new_input["expression"]

    write_h5(x=new_input, h5_filename=input_h5_filename)


@pytest.mark.local
def test_read_h5():
    from cfe.method.fate_dynverse_docker_backend import read_h5

    dynverse_docker_output = read_h5(h5_filename=output_h5_filename)

    # test trajectory result
    assert dynverse_docker_output.milestone_network is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])

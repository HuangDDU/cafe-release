import docker
import pytest

import cfe

from .test_fate_function_backend import get_test_run_data

image_id = "huangzhaoyang/cf_paga:0.0.1"


class TestCFEDockerBackend:
    def setup_method(self):
        self.cfe_docker = cfe.method.CFEDockerBackend(image_id)

    def test_init(self):
        assert self.cfe_docker.image_id == image_id

    def test_load_backend(self):
        # load_backend has benn called in __init__, implemented in DockerBackend
        assert self.cfe_docker.definition is not None

    def test_run(self):
        # TODO: image is not uploaded to docker hub
        fadata, parameters = get_test_run_data()
        self.cfe_docker.run(fadata, parameters)
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
        definition = self.cfe_docker.definition
        assert isinstance(definition, cfe.method.Definition)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

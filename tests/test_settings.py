import pytest

import cfe


def test_settings():
    backend = cfe.settings["backend"]
    assert backend in [None, "python_function", "cfe_docker", "dynverse_docker"]
    # assert backend in [None, "python_function", "conda", "cfe_docker", "dynverse_docker"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])

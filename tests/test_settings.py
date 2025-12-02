import pytest

import cafe


def test_settings():
    backend = cafe.settings["backend"]
    # assert backend in [None, "python_function", "cafe_docker", "dynverse_docker"]
    assert backend in [None, "python_function", "conda", "cafe_docker", "dynverse_docker"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])

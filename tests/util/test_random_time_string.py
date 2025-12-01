import re

import pytest

import cafe


def test_random_time_string_without_name():
    # time_string = cafe.util.random_time_string()
    name = ""
    time_string = cafe.util.random_time_string(name=name)

    assert re.match(r"\d{8}_\d{6}__", time_string), "time format wrong"

    parts = time_string.split("__")
    assert len(parts) == 2, "string split result length should be 2"
    assert len(parts[1]) == 10, "random string length should be 10"
    assert all(c.isalnum() for c in parts[1]), "invalid char in random string"


def test_random_time_string_with_name():
    name = "test_name"
    time_string = cafe.util.random_time_string(name=name)

    assert re.match(r"\d{8}_\d{6}__" + re.escape(name) + "__", time_string), "time format wrong"

    parts = time_string.split("__")
    assert len(parts) == 3, "string split result length should be 2"
    assert len(parts[2]) == 10, "random string length should be 10"
    assert all(c.isalnum() for c in parts[2]), "invalid char in random string"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

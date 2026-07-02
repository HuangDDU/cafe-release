from scripts.check_version import read_pyproject_version, validate_versions


def test_release_candidate_version_is_declared():
    assert read_pyproject_version() == "0.2.1rc1"


def test_pyproject_and_runtime_versions_are_consistent():
    assert validate_versions(expected="0.2.1rc1") == []

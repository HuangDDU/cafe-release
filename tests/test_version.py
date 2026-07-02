from scripts.check_version import check_version


def test_versions_are_consistent_and_pep440():
    assert check_version() == []


def test_expected_version_is_validated():
    assert check_version("not-a-version")

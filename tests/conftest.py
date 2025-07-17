import pytest


# You can add `-m run_method` in command to run this test markered as run_method. However, vscode automatical test will skip it.
def pytest_collection_modifyitems(config, items):
    for item in items:
        if "run_method" in item.keywords:
            if "-m run_method" not in " ".join(config.invocation_params.args):
                item.add_marker(pytest.mark.skip(reason="The test should run in specific conda environment with `-m run_method` in command"))

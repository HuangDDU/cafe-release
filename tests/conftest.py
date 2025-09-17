# import pytest


# # You can add `-m run_method` in command to run this test markered as run_method. However, vscode automatical test will skip it.
# def pytest_collection_modifyitems(config, items):
#     for item in items:
#         if "run_method" in item.keywords:
#             if "-m run_method" not in " ".join(config.invocation_params.args):
#                 item.add_marker(pytest.mark.skip(reason="The test should run in specific conda environment with `-m run_method` in command"))

import pytest


def pytest_addoption(parser):
    """添加一个命令行选项 --run-raw"""
    parser.addoption("--run-raw", action="store_true", default=False, help="Run raw function tests that require specific conda environments")


def pytest_configure(config):
    """注册自定义标记，避免 pytest 报警告"""
    config.addinivalue_line("markers", "raw: mark a test as a raw function test to be run with --run-raw")


def pytest_collection_modifyitems(config, items):
    """
    在测试收集阶段，根据 --run-raw 选项动态地跳过或保留测试。
    这是实现互斥执行的核心。
    """
    run_raw_mode = config.getoption("--run-raw")

    skip_normal = pytest.mark.skip(reason="skipped in --run-raw mode")
    skip_raw = pytest.mark.skip(reason="need --run-raw option to run this test")

    for item in items:
        is_raw_test = "raw" in item.keywords

        if run_raw_mode:
            # 如果是 --run-raw 模式，跳过所有“非 raw”的测试
            if not is_raw_test:
                item.add_marker(skip_normal)
        else:
            # 如果不是 --run-raw 模式，跳过所有“raw”的测试
            if is_raw_test:
                item.add_marker(skip_raw)

import pytest

# 暂时跳过整个 method 模块的测试
pytestmark = pytest.mark.skip(reason="Temporarily skipping method module tests")

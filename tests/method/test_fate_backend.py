import pytest


class TestBackend:
    """abstract, cannot be instantiated"""

    pass


if __name__ == "__main__":
    pytest.main(["-v", __file__])

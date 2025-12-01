import pandas as pd
import pytest


# only need to test the 'scan_method' function, other function in the file are called by this function.
def test_scan_method():
    from cafe.method.method_util import scan_method

    result = scan_method(return_type="dataframe")

    assert isinstance(result, pd.DataFrame)
    assert "comp1" in result.index
    assert "name" in result.columns


if __name__ == "__main__":
    pytest.main(["-v", __file__])

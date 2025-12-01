import pandas as pd
import pytest

import cafe


def test_get_available_method_df():
    method_df = cafe.util.get_available_method_df()
    assert isinstance(method_df, pd.DataFrame)


def get_available_dataset_df():
    # TODO:
    pass


if __name__ == "__main__":
    pytest.main(["-v", __file__])

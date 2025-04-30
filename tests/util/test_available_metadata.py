import pytest
import cfe
import pandas as pd


def test_get_available_method_df():
    method_df = cfe.util.get_available_method_df()
    assert isinstance(method_df, pd.DataFrame)


def get_available_dataset_df():
    # TODO:
    pass

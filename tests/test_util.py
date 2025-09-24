import numpy as np
import pandas as pd


def compare_dataframes(df1, df2, on_columns):
    """compare two dataframes ignoring  the order of rows"""

    # two dataframe are sorted based on the specified columns
    df1_sorted = df1.sort_values(by=on_columns).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=on_columns).reset_index(drop=True)

    # if they are equal
    return df1_sorted.equals(df2_sorted)


def compare_dataframes_closely(df1, df2, on_columns):
    # two dataframe are sorted based on the specified columns
    df1_sorted = df1.sort_values(by=on_columns).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=on_columns).reset_index(drop=True)

    for column in df1_sorted.columns:
        if pd.api.types.is_numeric_dtype(df1_sorted[column]):
            # value column
            if not np.allclose(df1_sorted[column].values, df2_sorted[column].values):
                return False
        else:
            # Non value column
            if not (df1_sorted[column] == df2_sorted[column]).all():
                return False
    return True


if __name__ == "__main__":
    # Demo for compare_dataframes
    data1 = {"id": ["A", "B", "C"], "value": [1, 2, 3]}
    data2 = {"id": ["C", "A", "B"], "value": [3, 1, 2]}
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    # compare
    result = compare_dataframes(df1, df2, on_columns=["id"])
    print("Two dataframe are equal:", result)

    # Demo for compare_dataframes
    data1 = {"id": ["A", "B", "C"], "value": [1.000005, 2, 3]}
    data2 = {"id": ["C", "A", "B"], "value": [3.00001, 1, 2.000009]}
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    # compare
    result = compare_dataframes_closely(df1, df2, on_columns=["id"])
    print("Two dataframe are equal closely:", result)

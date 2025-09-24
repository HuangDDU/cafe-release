import pandas as pd

# import scipy.sparse as sp


def expand_matrix(mat, rownames=None, colnames=None, fill=0):
    """
    扩展矩阵（以 DataFrame 形式提供），使其具有所需的行和列标签。
    对于原始矩阵中缺少的单元格，将使用指定的填充值进行填充。

    参数
        mat : pd.DataFrame
            要扩展的矩阵。
        rownames：列表形式，可选
            所需的行标签。 如果无，则使用 mat.index。
        colnames : 列表形式，可选
            所需的列名。 如果无，则使用 mat.columns。
        fill : 标量，可选
            用于填充缺失单元格的值（默认为 0）。

    Expand a matrix (provided as a DataFrame) so that it has the desired row and column labels.
    For cells missing in the original matrix, fill with the specified fill value.

    Parameters:
        mat : pd.DataFrame
            The matrix to expand.
        rownames : list-like, optional
            Desired row labels. If None, use mat.index.
        colnames : list-like, optional
            Desired column names. If None, use mat.columns.
        fill : scalar, optional
            Value to fill in missing cells (default is 0).

    Returns:
        pd.DataFrame: The expanded matrix with rows and columns as specified.
    """

    if rownames is None:
        rownames = mat.index
    if colnames is None:
        colnames = mat.columns

    # 构建一个新的 DataFrame，填充 fill 值
    newmat = pd.DataFrame(fill, index=rownames, columns=colnames)
    # 找出原矩阵和新矩阵中共同的行与列
    common_rows = mat.index.intersection(newmat.index)
    common_cols = mat.columns.intersection(newmat.columns)
    newmat.loc[common_rows, common_cols] = mat.loc[common_rows, common_cols]
    return newmat

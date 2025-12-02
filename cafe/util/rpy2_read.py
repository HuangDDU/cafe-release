# ref: https://github.com/dynverse/dynclipy/blob/master/dynclipy/read.py
import numpy as np
import rpy2.rinterface as rinterface
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from scipy.sparse import csc_matrix

# 各种装饰器定义了R到python的数据转换策略


@ro.conversion.rpy2py.register(rinterface.SexpS4)
def convert_sparse(obj):
    if "dgCMatrix" in obj.rclass:
        x = obj.do_slot("x")
        p = obj.do_slot("p")
        i = obj.do_slot("i")
        dim = obj.do_slot("Dim")

        # csr = csr_matrix((x, i, p), shape = dim[::-1]) #  这样做的csr后续还得转置成csc，麻烦，不如直接做
        csc = csc_matrix((x, i, p), shape=dim)

        index = ro.r["rownames"](obj)
        columns = ro.r["colnames"](obj)

        if isinstance(index, rinterface.NULLType):
            index = None
        if isinstance(columns, rinterface.NULLType):
            columns = None

        # return pd.DataFrame.sparse.from_spmatrix(csr.transpose(), index = index, columns = columns)
        # 此处返回字典格式
        return {
            "csc": csc,
            "cell_ids": index,
            "feature_ids": columns,
        }
    return obj


@ro.conversion.rpy2py.register(rinterface.ListSexpVector)
def convert_list(obj):
    # check if a non-empty list
    if not isinstance(obj, rinterface.NULLType) and len(obj) > 0:
        # check if named list
        if not isinstance(obj.names, rinterface.NULLType) and len(obj) == len(obj.names):
            x = {name: ro.conversion.rpy2py(obj[i]) for i, name in enumerate(obj.names)}
        else:
            x = [ro.conversion.rpy2py(obj[i]) for i in range(len(obj))]

        # check if dataframe
        if "data.frame" in obj.rclass:
            x = pandas2ri.rpy2py_dataframe(ro.vectors.DataFrame(obj))

        return x
    else:
        return {}


@ro.conversion.rpy2py.register(rinterface.StrSexpVector)
def convert_character(obj):
    if len(obj) == 1:
        return str(obj[0])
    else:
        return [str(x) for x in obj]


@ro.conversion.rpy2py.register(rinterface.IntSexpVector)
def convert_integer(obj):
    if len(obj) == 1:
        return int(obj[0])
    else:
        return [int(x) for x in obj]


@ro.conversion.rpy2py.register(rinterface.FloatSexpVector)
def convert_double(obj):
    if len(obj) == 1:
        return float(obj[0])
    else:
        if "matrix" in obj.rclass:
            cell_ids = ro.r["rownames"](obj)
            feature_ids = ro.r["colnames"](obj)
            matrix = np.array([float(x) for x in obj]).reshape((len(feature_ids), len(cell_ids))).T  # matrix in R is arranged by column
            matrix = csc_matrix(matrix)  # 转化为稀疏矩阵
            # return {
            #     "matrix":  matrix,
            #     "cell_ids": cell_ids,
            #     "feature_ids": feature_ids
            # }
            return matrix
        else:
            return [float(x) for x in obj]


@ro.conversion.rpy2py.register(rinterface.BoolSexpVector)
def convert_logical(obj):
    if len(obj) == 1:
        return bool(obj[0])
    else:
        return [bool(x) for x in obj]


# 对于空对象，直接返回


@ro.conversion.rpy2py.register(rinterface.NULLType)
def convert_null(obj):
    return None


# recursively check inside lists and dicts for items that were not converted into python objects


def check_conversion_rpy2py(obj, position="/", verbose=False):
    if verbose:
        print("✔ " + position)
    if isinstance(obj, dict):
        for name, obj2 in obj.items():
            check_conversion_rpy2py(obj2, position + name + "/")
    elif isinstance(obj, list):
        for i, obj2 in enumerate(obj):
            check_conversion_rpy2py(obj2, position + str(i) + "/")
    else:
        assert not isinstance(obj, rinterface.SexpVector), position + " could not be converted!"

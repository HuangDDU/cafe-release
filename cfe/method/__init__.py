# import importlib.util
# import os

from .fate_cfe_docker_backend import CFEDockerBackend
from .fate_conda_backend import CondaBackend
from .fate_dynverse_docker_backend import DynverseDockerBackend
from .fate_function_backend import FunctionBackend
from .fate_method import FateMethod
from .function import comp1
from .function.method_decorator import scan_method

__all__ = [
    "Definition",
    "FateMethod",
    "DynverseDockerBackend",
    "CFEDockerBackend",
    "FunctionBackend",
    "CondaBackend",
    #
    "comp1",
    "scan_method",
]

# directly import


# export many trajectory method.
# # static export is readable but code is tedious
# from .function.cf_paga import cf_paga
# from .function.cf_paga import paga as paga_raw
# # for pytest unit test
# __all__.append(cf_paga)
# # for easy call FateMethod object
# paga = FateMethod("paga") # create FateMethod object
# paga.__doc__ = paga_raw.__doc__ # binding detail parameter docs
# # comp1, scvelo and so on is needed

# (Suggeseted)dynamic export is scalable but is not readable and ignored by mkdocs
# function_name_list = [
#     "cf_paga",
#     "cf_comp1",
#     "cf_angle",
#     "cf_state_comp",
#     "cf_cluster_mst",
#     "cf_projection_mst",
#     "cf_graph_mst",
#     "cf_scvelo",
#     "cf_velovi",
#     "cf_palantir",
#     "cf_cytotrace2",
# ]
# for function_name in function_name_list:
#     function_file_path = f"{os.path.dirname(__file__)}/function/{function_name}.py"
#     # for pytest unit test, use cfe.method.cf_paga to run function directly
#     spec = importlib.util.spec_from_file_location(function_name, function_file_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)  # import module
#     module_function = getattr(module, function_name)  # import function
#     globals()[function_name] = module_function  # binding to global
#     __all__.append(function_name)
#     # for easy call FateMethod object, use 'cfe.method.paga' to run with multiple backend
#     easy_function_name = function_name[3:]
#     fate_method_obj = FateMethod(easy_function_name)  # create FateMethod object
#     # TODO: need export in mkdocs and merge docs in ..util.doc.merge_docs
#     fate_method_obj.__doc__ = getattr(module, easy_function_name).__doc__  # binding detail parameter docs,
#     globals()[easy_function_name] = fate_method_obj  # binding to global
#     __all__.append(function_name)

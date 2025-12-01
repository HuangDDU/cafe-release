# import functools
# import inspect
# from typing import Dict, List, Any, Callable
# import pandas as pd

# # store global method registry dict
# METHOD_REGISTRY = {}

# def method_info(
#     name: str,
#     version: str = "0.0.1",
#     description: str = "",
#     wrapper_type: str = "linear",
# ):
#     """
#     decorator for add meta infomation for trajectory method
#     """
#     def decorator(func: Callable) -> Callable:
#         # restore it in dict
#         METHOD_REGISTRY[name] = {
#             "function": func,
#             "name": name,
#             "version": version,
#             "description": description,
#             "wrapper_type": wrapper_type
#         }

#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             return func(*args, **kwargs)

#         return wrapper
#     return decorator


# def scan_method():
#     """
#     scan method and get docstring
#     """
#     method_dict = METHOD_REGISTRY.copy()
#     for name, metadata_dict in method_dict.keys():
#         method_dict["doc"] = metadata_dict["function"].__doc__
#     return pd.DataFrame(method_dict)


def scan_method():
    """
    TODO: scan method and get docstring
    """
    pass

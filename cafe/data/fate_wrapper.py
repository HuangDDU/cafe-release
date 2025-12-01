# import h5py
# import numpy as np


class FateWrapper:
    def __contains__(self, item):
        "check if have attribute"
        return hasattr(self, item)

    def keys(self):
        """return all attibute name, then the function dict() can be used"""
        return self.__dict__.keys()

    def __getitem__(self, key):
        "get attribute"
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    # def __write_h5ad__(self, group: h5py.Group):
    #     print(f"{self.__class__.__name__} __write_h5ad__")
    #     for key, value in self.__dict__.items():
    #         if isinstance(value, (str, int, float, bool)):
    #             group.attrs[key] = value
    #         elif isinstance(value, np.ndarray):
    #             group.create_dataset(key, data=value)
    #         # elif isinstance(value, pd.DataFrame):
    #         #     pass
    #         else:
    #             # TODO: Dataframe或其他类型强制转化为字符串，信息丢失
    #             group.create_dataset(key, data=str(value))
    #     # TODO：简化方式JSON

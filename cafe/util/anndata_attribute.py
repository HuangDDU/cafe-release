class AnndataAttribute:
    def __init__(self, adata):
        self.primary_attribute_dict = self.get_attribute_dict(adata)  # 注册旧的结构

    def get_attribute_dict(self, adata):
        # 2 level attribute dict for anndata attributes
        attribute_dict = {
            "obs": list(adata.obs.columns),
            "var": list(adata.var.columns),
            "obsm": list(adata.obsm.keys()),
            "varm": list(adata.varm.keys()),
            "obsp": list(adata.obsp.keys()),
            "varp": list(adata.varp.keys()),
            "layers": list(adata.layers.keys()),
            "uns": list(adata.uns.keys()),
        }
        return attribute_dict

    def get_external_attribute_dict(self, adata):
        new_attribute_dict = self.get_attribute_dict(adata)
        external_attribute_dict = {}
        for k, v in new_attribute_dict.items():
            external_attribute_dict[k] = list(set(v) - set(self.primary_attribute_dict[k]))
        return external_attribute_dict

    def extract_external_data_dict(self, adata):
        external_attribute_dict = self.get_external_attribute_dict(adata)
        external_data_dict = {}
        for k, v in external_attribute_dict.items():
            # k is like "obs", "var", etc for first level attribute name
            # v is list of attribute names for second level attribute name
            inner_external_data = {}
            for v_v in v:
                # v is external data
                inner_external_data[v_v] = getattr(adata, k)[v_v]
            external_data_dict[k] = inner_external_data
        return external_data_dict


def recovery_external_data(adata, external_data_dict):
    adata = adata.copy()
    for k, v in external_data_dict.items():
        for v_k, v_v in v.items():
            adata_attr = getattr(adata, k)
            if v_k in adata_attr:
                print(f"Warning: {v_k} in adata.{k} already exists and will Nnot overwritten.")
            else:
                adata_attr[v_k] = v_v
    return adata


def extract_external_data_dict_directly(adata, adata_new):
    # which is suitable for fate_conda_backend and fate_docker_backend. there are two adata objects
    anndata_attribute = AnndataAttribute(adata)
    external_data_dict = anndata_attribute.extract_external_data_dict(adata_new)
    return external_data_dict


# which is suitable for fate_function_backend, there is only one adata object
# anndata_attribute = AnndataAttribute(adata)
# ......
# external_data_dict = anndata_attribute.extract_external_data_dict(adata)

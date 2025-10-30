import os

import numpy as np
import pytest
import scanpy as sc

from cfe.util import AnndataAttribute


class TestAnndataAttribute:
    def setup_method(self):
        adata = sc.read_h5ad(f"{os.path.dirname(__file__)}/../data/bifurcating.h5ad")
        self.adata = adata
        self.anndata_attr = AnndataAttribute(adata)

        new_adata = adata.copy()
        n_obs, n_var = new_adata.shape
        external_data_dict = {
            "obs": {"new_obs_column": np.zeros(n_obs)},
            "var": {"new_var_column": np.zeros(n_var)},
            "obsm": {"new_obsm_key": np.zeros((n_obs, 2))},
            "varm": {"new_varm_key": np.zeros((n_var, 2))},
            "obsp": {"new_obsp_key": np.zeros((n_obs, n_obs))},
            "varp": {"new_varp_key": np.zeros((n_var, n_var))},
            "layers": {"new_layers_key": np.zeros((n_obs, n_var))},
            "uns": {"new_uns_key": 0},
        }
        for k, v in external_data_dict.items():
            getattr(adata, k).update(v)
        # new_adata.obs["new_obs_column"] = np.zeros(n_obs)
        # new_adata.var["new_var_column"] = np.zeros(n_var)
        # new_adata.obsm["new_obsm_key"] = np.zeros((n_obs, 2))
        # new_adata.varm["new_varm_key"] = np.zeros((n_var, 2))
        # new_adata.obsp["new_obsp_key"] = np.zeros((n_obs, n_obs))
        # new_adata.varp["new_varp_key"] = np.zeros((n_var, n_var))
        # new_adata.uns["new_uns_key"] = 0
        self.new_data = new_adata
        self.external_data_dict = external_data_dict

    def test_init(self):
        assert isinstance(self.anndata_attr.primary_attribute_dict, dict)

    def test_get_adata_attribute_dict(self):
        attr_dict = self.anndata_attr.get_attribute_dict(self.adata)
        assert isinstance(attr_dict, dict)

    def test_get_external_attribute_dict(self):
        external_attribute_dict = self.anndata_attr.get_external_attribute_dict(self.new_data)
        assert isinstance(external_attribute_dict, dict)

    def test_extract_external_data_dict(self):
        external_data_dict = self.anndata_attr.extract_external_data_dict(self.new_data)
        assert isinstance(external_data_dict, dict)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

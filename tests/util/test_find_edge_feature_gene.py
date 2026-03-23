import pytest

import cafe

from ..test_data import get_test_fadata


class TestFindEdgeFeatureGene:
    def setup_method(self):
        self.fadata = get_test_fadata()

    def test_find_edge_feature_gene(self):
        fadata = self.fadata
        edge_list = [("A", "B"), ("B", "C")]
        edge_dict = cafe.util.find_edge_feature_gene(fadata, edge_list=edge_list, top_n=10)
        assert isinstance(edge_dict, dict)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

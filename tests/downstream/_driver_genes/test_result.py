import pandas as pd
import pytest

import cafe
from cafe.downstream._driver_genes.result import DriverGeneResult


@pytest.fixture
def result():
    return DriverGeneResult(
        table=pd.DataFrame(
            {
                "gene": ["Gcg"],
                "lineage": ["Alpha"],
                "reference": ["Beta"],
                "score": [5.0],
                "logfoldchange": [2.0],
                "pval": [0.001],
                "qval": [0.01],
                "method": ["de"],
                "comparison": ["Alpha vs Beta"],
            }
        ),
        method="de",
        params={"test": "wilcoxon"},
        lineages=("Alpha", "Beta"),
        model_name="tree",
        metadata={"schema_version": 1, "expression_source": "X"},
    )


def test_result_persists_and_restores(fadata, result):
    before_obs = fadata.obs.copy(deep=True)
    result.write_to_fadata(fadata, key="markers")
    restored = DriverGeneResult.from_fadata(fadata, key="markers")
    pd.testing.assert_frame_equal(restored.table, result.table)
    assert restored.params == result.params
    assert restored.lineages == result.lineages
    pd.testing.assert_frame_equal(fadata.obs, before_obs)


def test_result_requires_explicit_overwrite(fadata, result):
    result.write_to_fadata(fadata, key="markers")
    with pytest.raises(KeyError, match="already exists"):
        result.write_to_fadata(fadata, key="markers")
    result.write_to_fadata(fadata, key="markers", overwrite=True)


def test_result_survives_h5ad_roundtrip(fadata, result, tmp_path):
    result.write_to_fadata(fadata, key="markers")
    path = tmp_path / "result.h5ad"
    fadata.write_h5ad(path)
    restored = DriverGeneResult.from_fadata(cafe.data.read_h5ad(path), key="markers")
    pd.testing.assert_frame_equal(restored.table, result.table)
    assert restored.metadata == result.metadata


def test_top_genes_ranks_and_filters_without_mutation(result):
    result.table = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D", "E"],
            "lineage": ["Alpha", "Alpha", "Alpha", "Beta", "Beta"],
            "score": [2.0, 5.0, -6.0, 4.0, -7.0],
            "logfoldchange": [1.0, 2.0, -3.0, 2.0, -2.0],
            "qval": [0.01, 0.2, 0.01, 0.02, 0.03],
        }
    )
    before = result.table.copy(deep=True)
    assert result.top_genes(n=1).gene.tolist() == ["B", "D"]
    assert result.top_genes(n=1, qval=0.05, direction="up").gene.tolist() == ["A", "D"]
    assert result.top_genes(n=1, direction="down").gene.tolist() == ["C", "E"]
    assert result.top_genes(qval=0.001).empty
    pd.testing.assert_frame_equal(result.table, before)

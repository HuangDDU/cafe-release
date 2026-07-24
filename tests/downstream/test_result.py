import numpy as np
import pandas as pd
import pytest

import cafe
from cafe.downstream._result import DriverGeneResult


def make_fadata():
    fadata = cafe.data.FateAnnData(
        X=np.zeros((2, 1)),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["gene"]),
    )
    fadata.add_model_name("result_test")
    fadata.add_trajectory(
        milestone_network=pd.DataFrame(
            {
                "from": ["Root"],
                "to": ["End"],
                "length": [1.0],
                "directed": [True],
            }
        ),
        progressions=pd.DataFrame(
            {
                "cell_id": ["a", "b"],
                "from": ["Root", "Root"],
                "to": ["Root", "End"],
                "percentage": [1.0, 1.0],
            }
        ),
        generate_color=False,
    )
    return fadata


def make_result():
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


def test_result_is_only_persisted_when_explicitly_requested():
    fadata = make_fadata()
    result = make_result()
    before_obs = fadata.obs.copy(deep=True)

    result.write_to_fadata(fadata, key="alpha_vs_beta")
    restored = DriverGeneResult.from_fadata(fadata, key="alpha_vs_beta")

    pd.testing.assert_frame_equal(restored.table, result.table)
    assert restored.params == result.params
    assert restored.lineages == result.lineages
    pd.testing.assert_frame_equal(fadata.obs, before_obs)


def test_result_refuses_accidental_overwrite():
    fadata = make_fadata()
    result = make_result()
    result.write_to_fadata(fadata, key="alpha_vs_beta")

    with pytest.raises(KeyError, match="already exists"):
        result.write_to_fadata(fadata, key="alpha_vs_beta")

    result.write_to_fadata(fadata, key="alpha_vs_beta", overwrite=True)


def test_result_survives_h5ad_roundtrip(tmp_path):
    fadata = make_fadata()
    result = make_result()
    result.write_to_fadata(fadata, key="alpha_vs_beta")
    path = tmp_path / "result.h5ad"

    fadata.write_h5ad(path)
    restored_fadata = cafe.data.read_h5ad(path)
    restored = DriverGeneResult.from_fadata(restored_fadata, key="alpha_vs_beta")

    pd.testing.assert_frame_equal(restored.table, result.table)
    assert restored.metadata == result.metadata

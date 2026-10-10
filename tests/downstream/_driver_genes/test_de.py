import numpy as np
import pandas as pd
import pytest

from cafe.downstream._driver_genes.methods.base import DriverContext
from cafe.downstream._driver_genes.methods.de import DEMethod
from cafe.downstream._lineage import extract_lineages


def make_context(fadata):
    lineage = extract_lineages(fadata, start_milestone="Root")
    return DriverContext(fadata, lineage, ("Alpha", "Beta"))


def test_de_finds_markers_and_excludes_shared_cells(fadata):
    context = make_context(fadata)
    original_x = fadata.X.copy()
    original_obs = fadata.obs.copy(deep=True)
    method = DEMethod()
    method.validate(context)
    result = method.compute(context)

    for lineage, gene in [("Alpha", "Gcg"), ("Beta", "Ins1")]:
        print(lineage)
        ranked = result.table.query("lineage == @lineage").sort_values("score", ascending=False)
        assert ranked.iloc[0]["gene"] == gene
    assert result.metadata["cell_counts"] == {"Alpha": 4, "Beta": 4}
    np.testing.assert_array_equal(fadata.X, original_x)
    pd.testing.assert_frame_equal(fadata.obs, original_obs)
    assert "rank_genes_groups" not in fadata.uns


def test_de_rejects_invalid_inputs(fadata):
    context = make_context(fadata)
    method = DEMethod()
    with pytest.raises(ValueError, match="at least 5 cells"):
        method.validate(context, min_cells=5)
    with pytest.raises(KeyError, match="layer"):
        method.validate(context, layer="missing")
    with pytest.raises(ValueError, match="use_raw"):
        method.validate(context, layer="counts", use_raw=True)
    with pytest.raises(ValueError, match="reference"):
        method.validate(context, reference="Missing")
    with pytest.raises(ValueError, match="test"):
        method.validate(context, test="logreg")

import numpy as np
import pandas as pd
import pytest

from cafe.downstream import driver_genes
from cafe.downstream._driver_genes.result import DriverGeneResult


def test_driver_genes_uses_active_model_and_prior_root(fadata):
    fadata.add_prior_information(start_milestone="Root")
    fadata.layers["lognorm"] = np.log1p(fadata.X)
    before_x = fadata.X.copy()
    before_obs = fadata.obs.copy(deep=True)
    result = driver_genes(fadata, layer="lognorm")

    assert result.model_name == "tree"
    assert result.lineages == ("Alpha", "Beta")
    assert result.metadata["cell_counts"] == {"Alpha": 4, "Beta": 4}
    assert "downstream" not in fadata.cafe_dict
    np.testing.assert_array_equal(fadata.X, before_x)
    pd.testing.assert_frame_equal(fadata.obs, before_obs)


def test_driver_genes_selects_lineages_and_reference(fadata):
    result = driver_genes(fadata, model_name="tree", start_cell="root", lineages=["Alpha", "Beta"], reference="Beta")
    assert set(result.table.lineage) == {"Alpha"}
    assert set(result.table.comparison) == {"Alpha vs Beta"}


def test_driver_genes_rejects_invalid_selection(fadata):
    with pytest.raises(KeyError, match="Unknown driver-gene method"):
        driver_genes(fadata, method="missing")
    with pytest.raises(KeyError, match="Unknown lineages"):
        driver_genes(fadata, start_milestone="Root", lineages=["Alpha", "Missing"])
    with pytest.raises(ValueError, match="unique"):
        driver_genes(fadata, start_milestone="Root", lineages=["Alpha", "Alpha"])


def test_driver_genes_preserves_saved_result_on_failure(fadata):
    saved = driver_genes(fadata, start_milestone="Root", key_added="markers")
    with pytest.raises(KeyError, match="already exists"):
        driver_genes(fadata, start_milestone="Root", key_added="markers")
    with pytest.raises(KeyError, match="layer"):
        driver_genes(fadata, start_milestone="Root", key_added="markers", overwrite=True, layer="missing")
    restored = DriverGeneResult.from_fadata(fadata, "markers")
    pd.testing.assert_frame_equal(saved.table, restored.table)

    replacement = driver_genes(fadata, start_milestone="Root", key_added="markers", overwrite=True, reference="Beta")
    assert set(replacement.table.lineage) == {"Alpha"}

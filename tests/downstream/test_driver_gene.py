import pandas as pd
import pytest

import cafe
from tests.downstream.test_de import make_de_fadata


def test_public_driver_gene_workflow_returns_and_caches_results():
    fadata = make_de_fadata()
    original_obs = fadata.obs.copy(deep=True)

    driver = cafe.downstream.DriverGene(fadata, model_name="tree")
    lineages = driver.compute_lineages(
        start_milestone="Root",
        terminal_states=["Alpha", "Beta"],
    )
    result = driver.compute(method="de", lineages=["Alpha", "Beta"])

    assert driver.lineages is lineages
    assert driver.results["de"] is result
    assert result.model_name == "tree"
    assert result.lineages == ("Alpha", "Beta")
    pd.testing.assert_frame_equal(fadata.obs, original_obs)
    assert "rank_genes_groups" not in fadata.uns


def test_driver_gene_requires_prepared_lineages():
    driver = cafe.downstream.DriverGene(make_de_fadata(), model_name="tree")

    with pytest.raises(RuntimeError, match="Compute lineages first"):
        driver.compute(method="de")


def test_driver_gene_rejects_unknown_lineage():
    driver = cafe.downstream.DriverGene(make_de_fadata(), model_name="tree")
    driver.compute_lineages(start_milestone="Root")

    with pytest.raises(KeyError, match="Unknown lineages"):
        driver.compute(method="de", lineages=["Alpha", "Missing"])

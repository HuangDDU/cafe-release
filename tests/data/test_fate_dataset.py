import os

import anndata as ad
import numpy as np
from scipy import sparse as sp

import cafe
from cafe._settings import CellFateExplorerConfig
from cafe.data import fate_dataset


def test_read_h5ad():
    fadata = cafe.data.read_h5ad(f"{os.path.dirname(__file__)}/bifurcating_fadata.h5ad")
    assert isinstance(fadata, cafe.data.FateAnnData)


def test_default_data_dir_is_local_data_directory(monkeypatch):
    monkeypatch.delenv("CAFE_DATA_DIR", raising=False)

    config = CellFateExplorerConfig()

    assert config.data_dir == "data"


def test_create_fadata_reads_with_backup_url(monkeypatch):
    adata = ad.AnnData(X=sp.csr_matrix([[1.0], [2.0]]))
    adata.layers["spliced"] = sp.csr_matrix([[1.0], [2.0]])
    adata.obs["clusters"] = ["A", "B"]
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0]])
    calls = []

    def fake_read(filename, *, backup_url, cache):
        calls.append((filename, backup_url, cache))
        return adata

    monkeypatch.setattr(fate_dataset.sc, "read", fake_read)

    fadata = fate_dataset._create_fadata_from_file(
        filename="data/example.h5ad",
        backup_url="https://example.org/example.h5ad",
        cluster="clusters",
        basis="X_umap",
        id="example",
        prior_information={"cluster": "clusters", "basis": "X_umap"},
    )

    assert isinstance(fadata, cafe.data.FateAnnData)
    assert calls == [("data/example.h5ad", "https://example.org/example.h5ad", True)]


def test_read_pancreas_supplies_scvelo_backup_url(monkeypatch, tmp_path):
    captured = {}
    sentinel = object()

    def fake_create_fadata_from_file(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(fate_dataset, "_create_fadata_from_file", fake_create_fadata_from_file)
    monkeypatch.setattr(fate_dataset.settings, "data_dir", str(tmp_path))

    result = fate_dataset.read_pancreas()

    assert result is sentinel
    assert captured["filename"] == str(tmp_path / "Pancreas" / "endocrinogenesis_day15.h5ad")
    assert captured["backup_url"] == ("https://github.com/theislab/scvelo_notebooks/raw/master/" "data/Pancreas/endocrinogenesis_day15.h5ad")

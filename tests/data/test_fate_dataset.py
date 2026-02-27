import cafe
import os
import pytest

def test_read_h5ad():
    fadata = cafe.data.read_h5ad(f"{os.path.dirname(__file__)}/bifurcating_fadata.h5ad")
    assert isinstance(fadata, cafe.data.FateAnnData)
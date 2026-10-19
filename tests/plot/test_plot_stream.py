import os

import matplotlib.pyplot as plt
import pytest

import cafe

from .test_plot_trajectory import get_plot_fadata as get_plot_fadata_old


def get_plot_fadata():
    fadata = get_plot_fadata_old()
    fadata.add_prior_information(start_cell="a", basis="X_umap")
    return fadata


def test_plot_stream_cell():
    fadata = get_plot_fadata()
    cafe.plot.plot_stream(fadata, mode="cell")
    plt.savefig(f"{os.path.dirname(__file__)}/img/test_plot_stream(cell).png")


def test_plot_stream_density():
    fadata = get_plot_fadata()
    cafe.plot.plot_stream(fadata, mode="density")
    plt.savefig(f"{os.path.dirname(__file__)}/img/test_plot_stream(density).png")


if __name__ == "__main__":
    pytest.main(["-v", __file__])

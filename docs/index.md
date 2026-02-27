# Cafe: An integrated platform for exploring cell fate

![Cafe Framework](img/logo_legend.png)

[![test](https://github.com/HuangDDU/CellFateExplorer/actions/workflows/test.yml/badge.svg)](https://github.com/HuangDDU/CellFateExplorer/actions/workflows/test.yml)
[![document](https://readthedocs.org/projects/cellfateexplorer/badge/?version=latest)](https://cellfateexplorer.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/cafe-release.svg)](https://pypi.org/project/cafe-release)
[![License](https://img.shields.io/github/license/HuangDDU/CellFateExplorer)](https://github.com/HuangDDU/CellFateExplorer/blob/main/LICENSE)

**Cafe (Cellular Fate Explorer)** is a modular framework to study cellular dynamics based on single-cell RNA-seq data. It provides an integrated platform for *inferring*, *visualizing* and *benchmarking* cell fate trajectories.

## Framework

![Cafe Framework](img/framework.png)

## Key Applications

- **Trajectory Inference**: Infer cell fate trajectories using various backend methods (including Python-based, Docker-based, and Conda-based backends).
- **Visualization**: Visualize trajectories, embeddings, pseudotime, and velocity fields with high-quality plots.
- **Benchmarking**: Compare different trajectory inference methods using comprehensive metrics including topology, cluster, and feature importance.
- **Data Management**: Manage complex single-cell data structures with `FateAnnData`, extending the capabilities of AnnData for trajectory analysis.

## Citation

If you use Cafe in your research, please cite:

```bibtex
@article{huang2025cellfateexplorer,
    title={CellFateExplorer: An integrated platform for exploring cell fate},
    author={Huang, Zhaoyang and Ma, Haonan and Peng, Yuchuan and Zhao, Chenguang and Yu, Liang},
    journal={bioRxiv},
    pages={2025--02},
    year={2025},
    publisher={Cold Spring Harbor Laboratory}
}
```

## Contributing

We actively encourage any contribution! To get started, please check out the [development document](./development_document/index.md).

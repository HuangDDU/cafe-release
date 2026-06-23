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

- **Data Management**: Manage complex single-cell data structures with `FateAnnData`, extending the capabilities of AnnData for trajectory analysis.
- **Trajectory Inference**: Infer cell fate trajectories using various backend methods.
- **Visualization**: Visualize trajectories, embeddings, pseudotime, and velocity fields.
- **Benchmarking**: Compare methods using comprehensive metrics (topology, cluster, feature importance, etc.).

## Documentation

| I want to... | Go to... |
|-------------|----------|
| Understand concepts & architecture | **[Introduction](introduction/index.md)** — cell fate prediction, data structures, wrappers, plots, metrics |
| Get hands-on code examples | **[Tutorial](tutorial/index.md)** — quick start, benchmark, visualization, cellxgene |
| Look up functions' API | **[API Reference](api/index.md)** — auto-generated from docstrings |
| Contribute to cafe | **[Development Guide](development_document/index.md)** — contribution guides & project schedule |

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

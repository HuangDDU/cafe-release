![Cafe Framework](./docs/img/logo_legend.png)

[![test](https://github.com/HuangDDU/CellFateExplorer/actions/workflows/test.yml/badge.svg)](https://github.com/HuangDDU/CellFateExplorer/actions/workflows/test.yml)
[![document](https://readthedocs.org/projects/cellfateexplorer/badge/?version=latest)](https://cellfateexplorer.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/cafe-release.svg)](https://pypi.org/project/cafe-release)
[![License](https://img.shields.io/github/license/HuangDDU/CellFateExplorer)](https://github.com/HuangDDU/CellFateExplorer/blob/main/LICENSE)


# Cafe: An integrated platform for exploring cell fate

**Cafe (Cellular Fate Explorer)** is a modular framework to study cellular dynamics based on single-cell RNA-seq data. It provides an integrated platform for *inferring*, *visualizing* and *benchmarking* cell fate trajectories.

## Framework

![Cafe Framework](./docs/img/framework.png)

## Key Applications

- **Trajectory Inference**: Infer cell fate trajectories using various backend methods (including Python-based, Docker-based, and Conda-based backends).
- **Visualization**: Visualize trajectories, embeddings, pseudotime, and velocity fields with high-quality plots.
- **Benchmarking**: Compare different trajectory inference methods using comprehensive metrics including topology, cluster, and feature importance.
- **Data Management**: Manage complex single-cell data structures with `FateAnnData`, extending the capabilities of AnnData for trajectory analysis.

## Documentation

Comprehensive documentation is available at [here](https://cafe-release.readthedocs.io/en/latest/), including installation, tutorial, API and so on.

## Project Schedule

- [x] Main framework code
- [x] Document construction: instruction, tutorial, API
- [x] [Data module](./docs/development_document/shedule/data.md): FateAnnData data structure, data collection.
- [x] [Methods module](./docs/development_document/shedule/method.md): 4 backends, trajectory methods for 8 wrappers.
- [ ] [Benchmark](./docs/development_document/shedule/benchmark.md): Comprehensive metrics and benchmarking.
- [x] [Plot](./docs/development_document/shedule/plot.md): Beautiful plotting capabilities.
- [ ] [Downstream analysis](./docs/development_document/shedule/downstream_analysis.md): Interactive web platform, driver gene, and GRN analysis.

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

# Welcome to Cell Fate Explorer🔍 Document

## Introduction

![framework](./img/framework.png)

**Cell Fate Explorer(cfe)** is a integration platform for *inferring*, *visualizing* and *benchmarking* cell fate trajectory for single-cell RNA-seq data.

## Key Concept

**FateAnnData**: Unified trajectory inference data structure base AnnData[@anndata], includes two types of key external data:

- **Milestone**: The key nodes in the trajectory of cell fate are milestones, and the milestone network is a simplification of the cell trajectory.
- **Waypoint**: Sampling points for cell fate and trajectories, can be used to simplify calculations and visualize trajectories in embedding space.

**FateMethod**: Unified trajectory inference method interface, includes [three types of backend(Users can select one of the calls during the execution process)](./shedule/method.md), we highly recommend **Python Function** or **CFE Docker** backend:

## Toc

- [Tutorial](./tutorial.md)
- [API document](./api.md)
- [Change log](./trajectory_methods.md)

# Benchmark Roadmap

Cafe benchmark development should produce more than a method ranking. The benchmark module should become a guideline system that tells users which method is suitable for which dataset and biological question.

## Goal

```mermaid
flowchart LR
    A["Benchmark datasets"] --> B["Method runs"]
    B --> C["Metric table"]
    C --> D["Report"]
    D --> E["Guideline / recommendation"]
```

The benchmark track should answer:

1. Which data conditions are required by each method?
2. Which metrics are appropriate for each biological task?
3. Which methods are stable, interpretable, and practical?

## Dataset Zoo

Each dataset should be documented with a data card.

| Priority | Dataset type | Purpose |
|----------|--------------|---------|
| P0 | Pancreas-like clean differentiation | quick benchmark and tutorial |
| P0 | hematopoiesis | branching lineage and terminal fate |
| P1 | embryogenesis / gastrulation | time-series and complex topology |
| P1 | neural differentiation | long developmental continuum |
| P2 | perturbation or reprogramming | downstream validation |

Required data-card fields:

- dataset id
- source and citation
- organism and tissue
- cell count and gene count
- available layers: counts, spliced, unspliced
- known labels: cell type, cluster, terminal fate, time
- valid benchmark tasks
- known caveats

## Metric Plan

| Metric family | Status | Notes |
|---------------|--------|-------|
| topology | P0 | isomorphic, edge flip, HIM |
| cluster / branch assignment | P0 | F1 branches, F1 milestones |
| pseudotime | P0 | Pearson / Spearman correlation |
| velocity | P1 | direction cosine, in-cluster coherence |
| feature importance | P1 | driver gene agreement |
| resource | P1 | time and memory |
| robustness | P2 | subsampling and bootstrap stability |
| perturbation consistency | P2 | requires perturbation ground truth |

## Benchmark Score

Cafe should avoid a single universal score, but a task-specific score can be useful:

\[
S_{m,t} = \sum_{k=1}^{K} w_{t,k} \cdot \operatorname{scale}(q_{m,k})
\]

where \(S_{m,t}\) is the score of method \(m\) for task \(t\), \(q_{m,k}\) is metric \(k\), and \(w_{t,k}\) is the task-specific weight.

Example:

| Task | High-weight metrics |
|------|---------------------|
| topology recovery | HIM, edge flip, isomorphic |
| pseudotime ordering | pseudotime correlation |
| terminal fate prediction | branch F1, fate probability calibration |
| velocity analysis | velocity coherence, direction consistency |

## Student Deliverables

Milestone 1:

- [ ] collect 5-10 candidate benchmark datasets
- [ ] write data cards
- [ ] run Pancreas with 3-5 methods
- [ ] produce `metric_table.csv`

Milestone 2:

- [ ] add benchmark report template
- [ ] add failure log and warning system
- [ ] define method recommendation rules
- [ ] connect benchmark outputs to Agent / Skill documentation

Milestone 3:

- [ ] run cross-dataset benchmark
- [ ] generate final figures
- [ ] write benchmark guideline page

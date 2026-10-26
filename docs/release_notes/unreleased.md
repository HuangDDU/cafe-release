## Unreleased <span class="cafe-release-date">In development</span>

**Added**

- Waddington-OT and moscot adapters with separate conda environments, cell-aligned transport maps, and a dyngen comparison notebook.
- The `timezone` strategy: Waterfall-style per-source-cell aggregation, cross-time cell-type milestones, and time-based cell progressions.
- `plot_sankey()` for visualizing raw temporal transport mass.
- Expanded visualization tutorial with categorized examples and trajectory preparation guidance.

**Changed**

- Moved time-trajectory construction into `_time_trajectory` strategies; the default `base` preserves existing behavior.
- Marked saved time-trajectory results with the `time` wrapper type.

**Fixed**

- Forwarded time-strategy selection and parameters through `add_trajectory_by_type()`.
- Fixed point-only progression conversion, including entirely unmatched time trajectories.
- Fixed misplaced milestone labels, repeated titles, and inherited node colors in multi-model comparisons.
- Isolated STREAM cell and density views and their temporary annotations/layouts from the input object.
- Preserved user/model pseudotime columns and categorical colors/order during STREAM plotting.

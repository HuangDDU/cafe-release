# ref: https://moscot.readthedocs.io/en/latest/notebooks/tutorials/200_temporal_problem.html
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    from cafe.method.function.method_decorator import method_info
    from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="moscot",
    version="0.0.1",
    description="Temporal optimal transport with moscot",
    wrapper_type="time",
    github_url="https://github.com/theislab/moscot",
    use_gpu=True,
    cpu_parallelization=True,
    available=True,
)
def moscot(
    adata: ad.AnnData,
    time_key: str = "time",
    cluster_key: str = None,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = None,
    prepare_kwargs: dict = None,
    solve_kwargs: dict = None,
    trajectory_kwargs: dict = None,
):
    """Preprocess -> TemporalProblem.prepare/solve -> extract time transports.

    Optional solver imports stay local so Cafe can inspect parameters outside
    this conda environment. Set repreprocess=False for prepared X. kwargs groups
    target the existing preprocessing, solver, and Cafe wrapper independently.
    """
    from moscot.problems.time import TemporalProblem

    # 1. preprocess
    adata = adata.copy()
    if repreprocess:
        adata = preprocess_pipeline(adata, **(repreprocess_kwargs or {}))

    # 2. execute method
    prepare = {"joint_attr": {"attr": "X", "tag": "point_cloud"}, **(prepare_kwargs or {})}
    problem = TemporalProblem(adata).prepare(time_key=time_key, **prepare)
    problem.solve(**(solve_kwargs or {}))

    # 3,4. extract and save results
    trajectory_dict = extract_trajectory_dict(problem, adata, time_key, cluster_key, trajectory_kwargs)
    return trajectory_dict


def extract_trajectory_dict(problem, adata, time_key, cluster_key, trajectory_kwargs=None):
    """Align solver rows/columns by cell names; retain unnormalized OT weights."""
    tmaps = {}
    for pair, solution in problem.solutions.items():
        subproblem = problem.problems[pair]
        source = adata.obs_names[adata.obs[time_key] == pair[0]]
        target = adata.obs_names[adata.obs[time_key] == pair[1]]
        transport = pd.DataFrame(np.asarray(solution.transport_matrix), index=subproblem.adata_src.obs_names, columns=subproblem.adata_tgt.obs_names)
        tmaps[pair] = sparse.csr_matrix(transport.loc[source, target].to_numpy(), dtype=float)
    trajectory_dict = {
        "wrapper_type": "time",
        "tmaps": tmaps,
        "time_key": time_key,
        "cluster_key": cluster_key,
        **(trajectory_kwargs or {}),
    }
    return trajectory_dict

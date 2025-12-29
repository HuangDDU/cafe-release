# import anndata as ad
# import numpy as np
# import pandas as pd

# try:
#     # for docker
#     from method_decorator import method_info
#     from preprocess_pipeline import preprocess_pipeline
# except ImportError:
#     # for completed cafe environment
#     from cafe.method.function.method_decorator import method_info
#     from cafe.method.function.preprocess_pipeline import preprocess_pipeline


# @method_info(
#     name="moscot",
#     version="0.0.1",
#     description="Moscot: Multi-omic single-cell optimal transport",
#     wrapper_type="time",
#     doi="10.1038/s41586-024-08453-2",
#     github_url="https://github.com/theislab/moscot",
# )
# def moscot(
#     adata: ad.AnnData,
#     time_key: str = "time",
#     cluster_key: str = "celltype",
#     epsilon: float = 1e-3,
#     scale_cost: str = "mean",
#     max_iterations: int = int(1e7),
#     flow_threshold: float = 0.05,
#     **kwargs,
# ):
#     """Run Moscot temporal optimal transport for trajectory inference.

#     Moscot solves optimal transport problems between consecutive time points,
#     computing cell-to-cell transition probabilities. This implementation aggregates
#     the cell-level transport matrices into cluster-level transitions for cafe trajectory.

#     Args:
#         adata: AnnData object with time point annotations.
#         time_key: Column name in obs for time points. Must be categorical.
#         cluster_key: Column name in obs for cell type/cluster annotations.
#         epsilon: Entropy regularization parameter for Sinkhorn algorithm.
#         scale_cost: How to scale the cost matrix ("mean", "max", etc.).
#         max_iterations: Maximum iterations for Sinkhorn algorithm.
#         flow_threshold: Minimum flow to include an edge in milestone_network.
#         **kwargs: Additional parameters passed to TemporalProblem.solve().

#     Returns:
#         dict: trajectory_dict with wrapper_type="time" and tmaps.

#     Example:
#         >>> import cafe
#         >>> fadata = cafe.data.read_erythroid_lineage()
#         >>> fadata.obs["time"] = fadata.obs["stage"].apply(lambda x: float(x[1:])).astype("category")
#         >>> method = cafe.method.FateMethod(method_name="moscot")
#         >>> method.infer_trajectory(fadata, parameters={"time_key": "time", "cluster_key": "celltype"})
#     """
#     from moscot.problems.time import TemporalProblem

#     # Ensure time_key is categorical
#     if not pd.api.types.is_categorical_dtype(adata.obs[time_key]):
#         adata.obs[time_key] = adata.obs[time_key].astype("category")

#     # Prepare and solve temporal problem
#     tp = TemporalProblem(adata)
#     tp = tp.prepare(time_key=time_key)
#     tp = tp.solve(
#         epsilon=epsilon,
#         scale_cost=scale_cost,
#         max_iterations=max_iterations,
#         **kwargs,
#     )

#     # Extract transport matrices from all time pairs
#     time_list = adata.obs[time_key].cat.categories.tolist()
#     time_pair_list = list(zip(time_list[:-1], time_list[1:]))

#     tmaps = {}
#     for t1, t2 in time_pair_list:
#         try:
#             prob = tp.problems[(t1, t2)]
#             # Convert JAX array to numpy if needed
#             tm = prob.solution.transport_matrix
#             if hasattr(tm, "__array__"):
#                 tm = np.array(tm)
#             tmaps[(t1, t2)] = tm
#         except Exception as e:
#             print(f"Warning: Failed to extract transport matrix for {t1}->{t2}: {e}")
#             continue

#     # Build trajectory dict for cafe
#     trajectory_dict = {
#         "wrapper_type": "time",
#         "tmaps": tmaps,
#         "time_key": time_key,
#         "cluster_key": cluster_key,
#         "flow_threshold": flow_threshold,
#         "normalize": True,
#         # Store Moscot-specific results for later analysis
#         "temporal_problem": tp,
#         "time_pair_list": time_pair_list,
#     }

#     return trajectory_dict

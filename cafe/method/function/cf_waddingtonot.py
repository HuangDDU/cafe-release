# ref: https://broadinstitute.github.io/wot/tutorial/

import anndata as ad
from scipy import sparse

try:
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    from cafe.method.function.method_decorator import method_info
    from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="Waddington-OT",
    version="0.0.1",
    description="Time-course optimal transport with WOT",
    wrapper_type="time",
    github_url="https://github.com/broadinstitute/wot",
    use_gpu=False,
    cpu_parallelization=True,
    available=True,
)
def waddingtonot(
    adata: ad.AnnData,
    time_key: str = "time",
    cluster_key: str = None,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = None,
    model_kwargs: dict = None,
    trajectory_kwargs: dict = None,
):
    """Preprocess -> OTModel -> in-memory adjacent-time transport extraction.

    model_kwargs forwards WOT model/solver options (local_pca, growth_iters,
    epsilon, etc.). No growth scores are invented. trajectory_kwargs controls
    Cafe's time wrapper; repreprocess=False uses an already prepared X.
    """
    from wot.ot import OTModel

    # 1. preprocess
    adata = adata.copy()
    if repreprocess:
        adata = preprocess_pipeline(adata, **(repreprocess_kwargs or {}))

    # 2. execute method
    model = OTModel(adata, day_field=time_key, **(model_kwargs or {}))
    times = model.timepoints
    maps = {pair: model.compute_transport_map(*pair) for pair in zip(times[:-1], times[1:])}

    # 3,4. extract and save results
    trajectory_dict = extract_trajectory_dict(maps, adata, time_key, cluster_key, trajectory_kwargs)
    return trajectory_dict


def extract_trajectory_dict(maps, adata, time_key, cluster_key, trajectory_kwargs=None):
    """Reorder WOT AnnData transports into input obs order, without pruning."""
    tmaps = {}
    for pair, transport in maps.items():
        source = adata.obs_names[adata.obs[time_key] == pair[0]]
        target = adata.obs_names[adata.obs[time_key] == pair[1]]
        tmaps[pair] = sparse.csr_matrix(transport[source, target].X, dtype=float)
    trajectory_dict = {
        "wrapper_type": "time",
        "tmaps": tmaps,
        "time_key": time_key,
        "cluster_key": cluster_key,
        **(trajectory_kwargs or {}),
    }
    return trajectory_dict

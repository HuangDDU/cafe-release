import anndata as ad
import scanpy as sc

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cafe environment
    from cafe.method.function.method_decorator import method_info
    from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="Comp1",
    version="0.0.1",
    description="Comp1: baseline for linear wrapper, extract an embedded component pseudotime method",
    wrapper_type="linear",
    use_gpu=False,
    cpu_parallelization=True,
    available=True,
)
def comp1(
    adata: ad.AnnData,
    repreprocess: bool = True,
    basis: str = "X_pca",
    recompute_basis: bool = False,
    component: int = 1,
) -> dict:
    """Comp1: baseline for linear wrapper, extract an embedded component pseudotime method

    Args:
        adata (ad.AnnData): The input AnnData object.
        repreprocess (bool, optional): Whether to preprocess the data.
        basis (str, optional): The embedding name in .obsm.
        recompute_basis (bool, optional): Whether to recompute the embedding.
        component (int, optional): The component number.

    Returns:
        dict: A trajectory dict of linear wrapper.
    """
    # 1. preprocess
    embedding_method = basis[2:].lower() if basis.startswith("X_") else basis.lower()
    if repreprocess and recompute_basis:
        # stop at sc.pp.pca, or sc.pp.neighbors if other embedding method
        preprocess_pipeline(adata, style="scanpy", if_neighbors=False if embedding_method == "pca" else True)

    # 2. execute method
    if recompute_basis or (basis not in adata.obsm):
        # execute dimension reduction if basis not in adata.obsm
        available_embedding_methods = ["pca", "tsne", "umap"]  # TODO: phate, diffmap ...
        # recompute the embedding
        if embedding_method in available_embedding_methods:
            if embedding_method == "pca":
                pass  # already computed in preprocess_pipeline
            elif embedding_method == "tsne":
                sc.tl.tsne(adata)
            elif embedding_method == "umap":
                sc.tl.umap(adata)
        else:
            # default use pca
            print(f"embedding method '{embedding_method}' is not available, use 'PCA' instead")
            basis = "pca"

    # 3. extract results
    pseudotime = adata.obsm[basis][:, component - 1].tolist()

    # 4. save results
    trajectory_dict = {
        "wrapper_type": "linear",
        "pseudotime": pseudotime,
    }

    return trajectory_dict

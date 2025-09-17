import anndata as ad
import scanpy as sc

try:
    from method_decorator import method_info
except ImportError:
    from cfe.method.function.method_decorator import method_info


@method_info(
    name="comp1",
    version="0.0.1",
    description="Comp1: baseline for linear wrapper, extract an embedded component pseudotime method",
    wrapper_type="linear",
)
def comp1(
    adata: ad.AnnData,
    repreprocess: bool = True,
    basis: str = "X_pca",
    component: int = 1,
) -> dict:
    """Comp1: baseline for linear wrapper, extract an embedded component pseudotime method

    Args:
        adata (ad.AnnData): The input AnnData object.
        repreprocess (bool, optional): Whether to preprocess the data.
        basis (str, optional): The embedding name in .obsm.
        component (int, optional): The component number.

    Returns:
        dict: A trajectory dict with keys: "wrapper_type" and "pseudotime".
    """
    # 1. preprocess
    if repreprocess:
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        adata = adata[:, adata.var["highly_variable"]]

    # 2. execute method
    if basis not in adata.obsm:
        # execute dimension reduction if basis not in adata.obsm
        embedding_method = basis[2:].lower()
        available_embedding_methods = ["pca", "tsne", "umap"]  # TODO: phate, diffmap ...
        # recompute the embedding
        if embedding_method in available_embedding_methods:
            if embedding_method == "pca":
                sc.pp.pca(adata)
            elif embedding_method == "tsne":
                sc.tl.tsne(adata)
            elif embedding_method == "umap":
                sc.pp.pca(adata)  # need pca first
                sc.pp.neighbors(adata)
                sc.tl.umap(adata)
        else:
            # default use pca
            print(f"embedding method '{embedding_method}' is not available, use 'PCA' instead")
            basis = "X_pca"
            sc.pp.pca(adata)

    # 3. extract results
    pseudotime = adata.obsm[basis][:, component - 1].tolist()

    # 4. save results
    trajectory_dict = {
        "wrapper_type": "linear",
        "pseudotime": pseudotime,
    }

    return trajectory_dict

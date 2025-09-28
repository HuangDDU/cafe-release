def preprocess_pipeline(adata, style="scanpy", **kwargs):
    if style == "scvelo":
        return scvelo_preprocess_pipeline(adata, **kwargs)
    elif style == "dynamo":
        return dynamo_preprocess_pipeline(adata, **kwargs)
    else:
        return scanpy_preprocess_pipeline(adata, **kwargs)


def scanpy_preprocess_pipeline(
    adata,
    if_log: bool = True,
    if_hvg: bool = True,
    if_pca: bool = True,
    if_neighbors: bool = True,
    target_sum: int = 1e4,
    flavor: str = "seurat",
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 15,
):
    """standard scanpy preprocess pipeline"""
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=target_sum)
    if not if_log:
        return adata
    sc.pp.log1p(adata)
    if not if_hvg:
        return adata
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
    if not if_pca:
        return adata
    sc.pp.pca(adata, n_comps=n_pcs)
    if not if_neighbors:
        return adata
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    return adata


def scvelo_preprocess_pipeline(
    adata,
    min_shared_counts: int = 20,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 15,
):
    """standard scvelo preprocess pipeline"""
    import scvelo as scv

    scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    return adata


def dynamo_preprocess_pipeline(
    adata,
    preprocessor_kwargs: dict = {},
    recipe: str = "monocle",
    reduceDimension_kwargs: dict = {},
    # n_top_genes: int = 1000,
    # fg_kwargs: dict = {'shared_count': 20},
    # n_pca_components: int = 30,
):
    """standard dynamo preprocess pipeline"""
    import dynamo as dyn
    from dynamo.preprocessing import Preprocessor

    preprocessor = Preprocessor(**preprocessor_kwargs)
    preprocessor.preprocess_adata(adata, recipe="monocle")
    dyn.tl.reduceDimension(adata, **reduceDimension_kwargs)
    # dyn.pp.recipe_monocle(adata, n_top_genes=n_top_genes, fg_kwargs=fg_kwargs)
    # dyn.tl.reduceDimension(adata, n_pca_components=n_pca_components)

    return adata

import anndata as ad
import scanpy as sc


def sctc(
    adata: ad.AnnData,
    repreprocess: bool = True,
    complexity_index_kwargs={},
):
    import sctc

    # TODO: preprocess
    if repreprocess:
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var["highly_variable"]]

    cci, gci = sctc.complexity_index(adata.X.toarray())  # only use cci for cell
    pseudotime = (1 - cci).tolist()  # pseudotime and cci are negatively correlated

    trajectory_dict = {
        "wrapper_type": "linear",
        "pseudotime": pseudotime,
    }
    return trajectory_dict

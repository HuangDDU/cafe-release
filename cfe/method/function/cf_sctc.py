import anndata as ad
import scanpy as sc

try:
    from method_decorator import method_info
except ImportError:
    from cfe.method.function.method_decorator import method_info


@method_info(name="sctc", version="0.0.1", description="SCTC: single-Cell Transcriptional Complexity", wrapper_type="linear")
def sctc(
    adata: ad.AnnData,
    repreprocess: bool = True,
) -> dict:
    """SCTC: Single-Cell Transcriptional Complexity.

    Args:
        adata (ad.AnnData): The input AnnData object.
        repreprocess (bool, optional): Whether to preprocess the data. Defaults to True.

    Returns:
        dict: A trajectory dict with keys: "wrapper_type" and "pseudotime".
    """

    import sctc

    # 1. preprocess
    if repreprocess:
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        adata = adata[:, adata.var["highly_variable"]]

    # 2. execute method
    cci, gci = sctc.complexity_index(adata.X.toarray())  # only use cci for cell

    # 3. extract results
    pseudotime = (1 - cci).tolist()  # pseudotime and cci are negatively correlated

    # 4. save results
    trajectory_dict = {
        "wrapper_type": "linear",
        "pseudotime": pseudotime,
    }
    return trajectory_dict

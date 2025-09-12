import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def cytotrace2(adata: ad.AnnData, repreprocess: bool = True, cluster_key: str = None, cytotrace2_kwargs: dict = {}, **kwargs):
    from cytotrace2_py.cytotrace2_py import cytotrace2

    with tempfile.TemporaryDirectory() as tmp_wd:
        # 1. preprocess:
        if repreprocess:
            # cytotrace2 don't recommend use log-transformed expression matrix and HVGs, only normalized here.
            sc.pp.normalize_per_cell(adata)
        else:
            if not (np.issubdtype(adata.X.dtype, np.integer) or np.isclose(adata.X.data, np.round(adata.X.data)).all()):
                print("warnning: raw expression matrix is transformed, count matrix is not available,")
            else:
                print("use count matrix")
        X = adata.X.toarray()
        # write tmp file: expression matrix and annotation(if required)
        expression_file = f"{tmp_wd}/cytotrace2_expression.csv"
        annotation_path = ""
        df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names).T
        print(f"write expression matrix({df.shape}) to  {expression_file}")
        df.to_csv(expression_file, sep="\t")
        if cluster_key is not None:
            annotation_path = f"{tmp_wd}/cytotrace2_annotations.csv"
            adata.obs[cluster_key].to_csv(annotation_path, sep="\t")
            print(f"write annotation to {annotation_path}")

        # 2. execute method
        result = cytotrace2(expression_file, annotation_path, disable_plotting=True, **cytotrace2_kwargs)

        # 3. extract results
        pseudotime = (1 - result["CytoTRACE2_Score"]).tolist()

        # 4. save results
        trajectory_dict = {
            "wrapper_type": "linear",
            "pseudotime": pseudotime,
        }

    return trajectory_dict

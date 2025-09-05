import tempfile

import anndata as ad
import pandas as pd


def cytotrace2(adata: ad.AnnData, repreprocess: bool = True, cluster_key: str = None, cytotrace2_kwargs: dict = {}, **kwargs):
    from cytotrace2_py.cytotrace2_py import cytotrace2

    with tempfile.TemporaryDirectory() as tmp_wd:
        # 1. preprocess: write tmp file: expression matrix and annotation(if required)
        expression_file = f"{tmp_wd}/cytotrace2_expression.csv"
        annotation_path = ""
        # TODO: extract count/normalized expression matrix, not log-transformed matrix
        df = pd.DataFrame(adata.X.toarray().astype(int), index=adata.obs_names, columns=adata.var_names).T
        print(
            f"write expression matrix({df.shape}) to  {expression_file}",
        )
        df.to_csv(expression_file, sep="\t")
        if cluster_key is not None:
            annotation_path = f"{tmp_wd}/cytotrace2_annotations.csv"
            adata.obs[cluster_key].to_csv(annotation_path, sep="\t")
            print(
                f"write annotation to {annotation_path}",
            )

        # 2. execute method
        result = cytotrace2(expression_file, annotation_path, disable_plotting=True, **cytotrace2_kwargs)
        pseudotime = (1 - result["CytoTRACE2_Score"]).tolist()

        # 3,4. extract and save results
        trajectory_dict = {
            "pseudotime": pseudotime,
        }

        return trajectory_dict


def cf_cytotrace2(
    adata: ad.AnnData,
    prior_information: dict = None,
    parameters: dict = None,
    **kwargs,
):
    if (prior_information is None) and (parameters is None):
        # for new backend call, function(**kwargs)
        return cytotrace2(adata, **kwargs)
    else:
        # for old backend call, function(prior_information, parameters)
        parameters.update(prior_information)
        return cytotrace2(adata, **parameters)

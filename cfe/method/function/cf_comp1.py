import anndata as ad
import scanpy as sc


def cf_comp1(
    adata: ad.AnnData,
    prior_information: dict = {},
    parameters: dict = {}
):
    # 1. prepare data
    adata = adata.copy()

    # 2. preprocess and execute method simutaneously with pca
    sc.pp.pca(adata, n_comps=parameters["ndim"])

    # 3. extract results
    pseudotime = adata.obsm["X_pca"][:, parameters["component"] - 1]

    # 4. save results
    trajectory_dict = {
        "pseudotime": pseudotime,
    }

    return trajectory_dict


if __name__ == "__main__":
    import pickle
    from parse_args import parse_args

    adata, prior_information, parameters, output_filename = parse_args()
    trajectory_dict = cf_comp1(adata, prior_information, parameters)
    with open(output_filename, "wb") as f:
        pickle.dump(trajectory_dict, f)
    print("Comp1 Finish!")

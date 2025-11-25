import anndata as ad
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler, normalize

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cfe environment
    from cfe.method.function.method_decorator import method_info
    from cfe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="state_comp",
    version="0.0.1",
    description="State_Comp: baseline for probability and lineage wrapper, state transition probability based on embedded components",
    wrapper_type="probability",
)
def state_comp(
    adata: ad.AnnData,
    repreprocess: bool = True,
    n_comps: int = 2,
    basis: str = "X_pca",
    recompute_basis: bool = False,
    pseudotime_index: int = 1,
    wrapper_type: str = "probability",  # "probability" or "lineage"
    cluster_key: str = "clusters",
):
    """State_Comp: baseline for probability and lineage wrapper, state transition probability based on embedded components

    Args:
         adata (ad.AnnData): The input AnnData object.
        repreprocess (bool, optional): Whether to preprocess the data.
        n_comps (int, optional): The number of components.
        basis (str, optional): The embedding name in .obsm.
        recompute_basis (bool, optional): Whether to recompute the embedding.
        pseudotime_index (int, optional): The index of the component to use for pseudotime.
        wrapper_type (str, optional): The type of wrapper to use.

    Returns:
        dict: A trajectory dict of probability or lineage wrapper.

    """
    # 1. preprocess
    embedding_method = basis[2:].lower() if basis.startswith("X_") else basis.lower()
    if repreprocess and recompute_basis:
        preprocess_pipeline(adata, style="scanpy", if_neighbors=False if basis == "X_pca" else True)  # stop as sc.pp.pca
    cell_ids = adata.obs.index

    # 2. execute method
    # like comp1 method, but extract multiple components as multiple end states
    if (basis not in adata.obsm) or recompute_basis:
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
    # extract embedding results as state transition probabilities
    X_emb = adata.obsm[basis][:, :n_comps]
    X_emb_scaled = MinMaxScaler().fit_transform(X_emb)  # Normalization
    comp_column_list = [f"comp_{i}" for i in range(1, n_comps + 1)]  # the first ndim components correspond to n states
    # The normalized PCA result is used as the state transition probability, range of [0,1]
    end_state_probabilities = pd.DataFrame(
        columns=comp_column_list,
        data=normalize(X_emb_scaled, norm="l1"),  # l1 transform
        index=cell_ids,
    )
    end_state_probabilities["cell_id"] = cell_ids
    end_state_probabilities = end_state_probabilities[["cell_id"] + comp_column_list]

    # 3,4. extract and save results
    if wrapper_type == "lineage":
        # for lineage wrapper
        trajectory_dict = {
            "probability": end_state_probabilities[end_state_probabilities.columns[1:]],
            "cluster_key": cluster_key,
        }
    else:
        # for probability wrapper
        pseudotime = X_emb_scaled[:, pseudotime_index]  # specified component for pseudotime
        trajectory_dict = {
            "end_state_probabilities": end_state_probabilities,
            "pseudotime": pseudotime,
        }

    trajectory_dict["wrapper_type"] = wrapper_type
    return trajectory_dict

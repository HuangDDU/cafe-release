DEFAULT_MODE = "embedding"


def _get_probability_df(fadata, model_name: str = None):
    raw_wrapper_dict = fadata.get_trajectory_dict(model_name=model_name)["raw_wrapper_dict"]
    # probability key: end_state_probabilities for dynverse result, probability for cfe result
    possible_keys = ["end_state_probabilities", "probability"]
    probability_df = None
    for key in possible_keys:
        if key in raw_wrapper_dict:
            probability_df = raw_wrapper_dict[key]
    if probability_df is None:
        raise ValueError(f"No probability data found in raw_wrapper_dict for keys: {possible_keys}")

    if "cell_id" in probability_df.columns:
        probability_df = probability_df.drop(columns=["cell_id"])
    return probability_df


def plot_embedding(fadata, model_name: str = None, basis=None):
    import scanpy as sc

    if basis is None:
        basis = fadata.prior_information.get("basis")

    probability_df = _get_probability_df(fadata, model_name=model_name)
    end_state_list = probability_df.columns.tolist()
    fadata.obs[end_state_list] = probability_df.values
    sc.pl.embedding(fadata, color=end_state_list, basis=basis)
    fadata.obs.drop(columns=end_state_list, inplace=True)


def plot_star(fadata, model_name: str = None):
    import cellrank as cr
    import pandas as pd

    probability_df = _get_probability_df(fadata, model_name=model_name)
    terminal_state_list = probability_df.columns.tolist()
    cluster_color_dict = dict(zip(fadata.obs["clusters"].cat.categories.tolist(), fadata.uns["clusters_colors"]))
    fadata.obs["term_states_fwd"] = pd.Categorical(probability_df.idxmax(axis=1), categories=terminal_state_list)
    fadata.uns["term_states_fwd_colors"] = [cluster_color_dict[t] for t in terminal_state_list]
    fadata.obsm["lineages_fwd"] = probability_df.values
    cr.pl.circular_projection(fadata, keys=["clusters"], legend_loc="right")

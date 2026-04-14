import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

from ..._settings import settings

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
    ax = sc.pl.embedding(fadata, color=end_state_list, basis=basis, show=False)
    fadata.obs.drop(columns=end_state_list, inplace=True)

    return ax


def plot_star(fadata, model_name: str = None, terminal_palette=settings.sns_palette):
    import cellrank as cr
    import pandas as pd

    probability_df = _get_probability_df(fadata, model_name=model_name)
    terminal_state_list = probability_df.columns.tolist()
    if len(terminal_state_list) <= 2:
        print("Warning: Less than 3 terminal states detected, star plot may not be informative.")

    # color for terminal states
    cluster_list = fadata.obs["clusters"].cat.categories.tolist()
    cluster_color_list = fadata.uns.get("clusters_colors", [])
    is_subset = all(state in cluster_list for state in terminal_state_list)
    if is_subset and len(cluster_color_list) > 0:
        # color from existing cluster color dict
        cluster_color_dict = dict(zip(cluster_list, cluster_color_list))
        terminal_state_color_list = [cluster_color_dict[t] for t in terminal_state_list]
    else:
        # new color dict
        palette = sns.color_palette(terminal_palette, n_colors=len(terminal_state_list))
        terminal_state_color_list = [mcolors.to_hex(color) for color in palette]

    # add cellrank related dadta
    fadata.obs["term_states_fwd"] = pd.Categorical(probability_df.idxmax(axis=1), categories=terminal_state_list)
    fadata.uns["term_states_fwd_colors"] = terminal_state_color_list
    fadata.obsm["lineages_fwd"] = probability_df.values

    # plot
    ax = cr.pl.circular_projection(fadata, keys=["clusters"], legend_loc="right")

    if ax is None:
        ax = plt.gca()

    return ax

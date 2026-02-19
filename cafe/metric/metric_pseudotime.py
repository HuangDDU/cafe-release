from scipy import stats

from ..data import FateAnnData


# TODO: add docs
def calculate_pseudotime_correlation(
    fadata: FateAnnData,
    ref_start_milestone=None,
    pred_start_milestone=None,
    start_cell=None,  # different trajectory use same cell as start cell
    ref_model: str = "ref",
    pred_model: str = "default",
):
    # get pusedotime for ref and pred model
    ref_pseudotime = fadata.get_trajectory_pseudotime(start_milestone=ref_start_milestone, start_cell=start_cell, model_name=ref_model)
    pred_pseudotime = fadata.get_trajectory_pseudotime(start_milestone=pred_start_milestone, start_cell=start_cell, model_name=pred_model)

    # calc correlation
    corr = stats.spearmanr(ref_pseudotime, pred_pseudotime).statistic
    return corr

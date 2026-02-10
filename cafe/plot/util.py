import matplotlib.pyplot as plt

from .. import logger


def save_fig(
    save: bool | str = None,
    default_filename: str = None,
    ax: plt.Axes = None,
):
    # save the figure to the specified path
    if save is not None:
        if isinstance(save, bool) and save:
            save = default_filename
        if ax is None:
            plt.savefig(save, bbox_inches="tight")
        else:
            ax.figure.savefig(save, bbox_inches="tight")
        logger.debug(f"save plot to '{save}'")
    pass

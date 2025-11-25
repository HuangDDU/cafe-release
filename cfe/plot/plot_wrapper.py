import matplotlib.pyplot as plt

from .._logging import logger
from ..data import FateAnnData
from ._wrapper_plots import PLOTTER_MODULE_REGISTRY


def plot_wrapper(fadata: FateAnnData, wrapper_type: str = None, model_name: str = None, mode: str = None, save: bool | str = None, **kwargs) -> None:
    """plot original wrapper data

    Args:
        fadata (FateAnnData): FateAnnData object
        wrapper_type (str, optional): wrapper type determines the plot style. Defaults to None.
    """
    if model_name is None:
        model_name = fadata.model_name

    # --- 1. Infer wrapper_type if not provided ---
    if wrapper_type is None:
        # extract wrapper type from fadata
        wrapper_type = fadata.get_raw_wrapper_dict(model_name).get("wrapper_type", "direct")
        logger.info(f"find wrapper type: {wrapper_type}")

    # --- 2. Find the correct plotter module ---
    plotter_module = PLOTTER_MODULE_REGISTRY.get(wrapper_type)
    if not plotter_module:
        logger.warning(f"No plotter module found for wrapper type '{wrapper_type}'. Nothing to plot.")
        return

    # --- 3. Determine the plot mode ---
    if mode is None:
        # Use the default mode defined in the module, or fallback to a common default
        mode = getattr(plotter_module, "DEFAULT_MODE", "embedding")
        logger.info(f"No mode specified, using default for '{wrapper_type}': '{mode}'")

    function_name = f"plot_{mode}"
    plot_function = getattr(plotter_module, function_name, None)  # Note: 核心函数

    if not plot_function:
        logger.error(f"Plotting mode '{mode}' (function '{function_name}') not found in module for wrapper '{wrapper_type}'.")
        # You could list available styles here for better user feedback
        available_modes = [s.replace("plot_", "") for s in dir(plotter_module) if s.startswith("plot_")]
        logger.info(f"Available modes for '{wrapper_type}': {available_modes}")
        return

    # --- 4. Dispatch to the specific plot function ---
    logger.debug(f"Dispatching to plotter '{wrapper_type}' with mode '{mode}'.")
    plot_function(fadata=fadata, model_name=model_name, **kwargs)

    # --- 5. Save the figure if requested ---
    if save is not None:
        if isinstance(save, bool) and save:
            save = f".cfe/{fadata.id}/img/wrapper_{model_name}.png"
        plt.savefig(save)
        logger.debug(f"save trajectory plot to '{save}'")

import sys

from ._logging import logger
from ._settings import settings

from . import data, method, metric, plot, preprocess, util, benchmark  # isort: skip

logo = """
    ██████╗ █████╗ ███████╗███████╗
  ██╔════╝██╔══██╗██╔════╝██╔════╝
  ██║     ███████║█████╗  █████╗  
  ██║     ██╔══██║██╔══╝  ██╔══╝  
  ╚██████╗██║  ██║██║     ███████╗
    ╚═════╝╚═╝  ╚═╝╚═╝     ╚══════╝
"""
logger.info(logo, indent_level=0)
logger.info(f"Version: {settings.version}", indent_level=0)

__all__ = ["settings", "logger", "data", "preprocess", "method", "plot", "util", "metric", "benchmark"]


# Compatible with 'cfe' module references in older version pickle files
sys.modules["cfe"] = sys.modules[__name__]
sys.modules["cfe.data"] = sys.modules["cafe.data"]
sys.modules["cfe.data.fate_anndata"] = sys.modules["cafe.data.fate_anndata"]
sys.modules["cfe.data.fate_milestone_wrapper"] = sys.modules["cafe.data.fate_milestone_wrapper"]
sys.modules["cfe.data.fate_waypoint_wrapper"] = sys.modules["cafe.data.fate_waypoint_wrapper"]

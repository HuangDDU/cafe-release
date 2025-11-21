from ._logging import logger
from ._settings import settings

from . import data, method, metric, plot, preprocess, util  # isort: skip

logo = """
   _____     _ _ ______    _       ______            _
  / ____|   | | |  ____|  | |     |  ____|          | |
 | |     ___| | | |__ __ _| |_ ___| |__  __  ___ __ | | ___  _ __ ___ _ __
 | |    / _ \\ | |  __/ _` | __/ _ \\  __| \\ \\/ / '_ \\| |/ _ \\| '__/ _ \\ '__|
 | |___|  __/ | | | | (_| | ||  __/ |____ >  <| |_) | | (_) | | |  __/ |
  \\_____\\___|_|_|_|  \\__,_|\\__\\___|______/_/\\_\\ .__/|_|\\___/|_|  \\___|_|
                                              | |
                                              |_|
"""
logger.info(logo, indent_level=0)
logger.info(f"Version: {settings.version}", indent_level=0)

__all__ = ["settings", "logger", "data", "preprocess", "method", "plot", "util", "metric"]

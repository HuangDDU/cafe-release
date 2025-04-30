from ._settings import settings
from ._logging import logger

from . import data
from . import preprocess
from . import method
from . import plot
from . import util
from . import metric


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
logger.info(logo)

__all__ = [
    "settings",
    "logger",
    "data",
    "preprocess",
    "method",
    "plot",
    "util",
    "metric"
]

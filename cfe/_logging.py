import logging

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["logger"]


def _setup_logger() -> "logging.Logger":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console = Console(width=200, force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False
    handler = RichHandler(log_time_format="[%m/%d/%Y %I:%M:%S]", show_path=False, console=console)
    logger.addHandler(handler)

    # this prevents double outputs
    logger.propagate = False
    return logger


logger = _setup_logger()


def format_logger(format):
    # reset format
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter(format))


if __name__ == "__main__":
    logger.info("Info Message")
    logger.debug("Debug Message")
    # try to modify the format
    format_logger("%(name)s -%(filename)s:%(lineno)d - %(message)s")
    logger.info("Info Message")

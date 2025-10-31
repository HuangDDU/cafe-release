import logging

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["logger", "set_log_file"]


def _setup_logger() -> "logging.Logger":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # console logger
    console_handler = Console(width=200, force_terminal=True)
    if console_handler.is_jupyter is True:
        console_handler.is_jupyter = False
    console_handler = RichHandler(log_time_format="[%m/%d/%Y %I:%M:%S]", show_path=False, console=console_handler)
    logger.addHandler(console_handler)
    # file logger
    set_log_file(".cfe/cfe_debug.log", logger)  # default log file

    logger.propagate = False  # this prevents double outputs
    return logger


def set_log_file(filename=".cfe/cfe_debug.log", logger=None):
    # set log file need to remove old file handler and create new file handler
    if logger is None:
        logger = globals().get("logger") or logging.getLogger(__name__)

    # remove old file handler
    old_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            old_handler = handler
            break

    if old_handler:
        logger.debug(f"set new log file: {filename}")
        old_handler.close()
        logger.removeHandler(old_handler)

    file_handler = logging.FileHandler(filename, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    # file_formatter = RichHandler(log_time_format="[%m/%d/%Y %I:%M:%S]", show_path=False, console=file_handler)
    logger.addHandler(file_handler)
    if old_handler:
        logger.debug(f"continue logging from old file: {old_handler.baseFilename}")


logger = _setup_logger()


def print_output(print=logger.info, output_list=[]):
    # read from command executed by subprocess and print output
    def fun(
        pipe,
        prefix,
    ):
        """print output from a pipe"""
        for line in iter(pipe.readline, ""):
            if line:
                print(f"{prefix}{line.rstrip()}")
                if output_list is not None:
                    output_list.append(line.rstrip())
        pipe.close()

    return fun


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

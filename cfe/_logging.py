import logging

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["logger", "set_log_file"]
mode = "dev"  # dev, test, prod


def _setup_logger() -> "logging.Logger":
    logger = logging.getLogger("cfe")
    logger.setLevel(logging.INFO)
    # console logger
    console_handler = Console(width=200, force_terminal=True)
    if console_handler.is_jupyter is True:
        console_handler.is_jupyter = False
    if mode == "dev":
        # int development stage, too much time info is tedious
        console_handler = RichHandler(show_time=False, show_path=False, console=console_handler)
    elif mode == "test":
        # test or benchmark need time
        console_handler = RichHandler(log_time_format="[%m/%d/%Y %I:%M:%S]", show_path=False, console=console_handler)
    else:
        # production simple output
        console_handler = RichHandler(show_time=False, show_path=False, show_level=False, console=console_handler)
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
    logger.addHandler(file_handler)
    if old_handler:
        logger.debug(f"continue logging from old file: {old_handler.baseFilename}")


def _format_logging_message(msg: str, level: int, indent_level: int = 0, indent_space_num: int = 4) -> str:
    """logging with level ident in dynamo style, ref: https://github.com/aristoteleo/dynamo-release/blob/master/dynamo/dynamo_logger.py"""

    indent = "-" * indent_space_num
    prefix = indent * indent_level
    # make it prettier: start with '|' and replace first char with it
    if prefix:
        prefix = "|" + prefix[1:]
    # # marker per level
    # if level == logging.INFO:
    #     marker = ">"
    # elif level == logging.WARNING:
    #     marker = "?"
    # elif level == logging.CRITICAL:
    #     marker = "!!"
    # elif level == logging.DEBUG:
    #     marker = ">>>"
    # elif level == logging.ERROR:
    #     marker = "!!"
    # else:
    #     marker = ">"
    # return f"{prefix}{marker} {msg}"
    return f"{prefix} {msg}"


class CFELogger:
    """Lightweight adapter around a standard logging.Logger that supports indent_level."""

    # when using, Only consider the hierarchical structure within the function, without considering the calling relationships between functions
    def __init__(self, base_logger: logging.Logger, indent_space_num: int = 2):
        self._base = base_logger
        self.indent_space_num = indent_space_num
        self.current_indent = 1

    # preserve basic logger interface used elsewhere
    def setLevel(self, *args, **kwargs):
        return self._base.setLevel(*args, **kwargs)

    @property
    def handlers(self):
        return self._base.handlers

    def addHandler(self, handler):
        return self._base.addHandler(handler)

    def removeHandler(self, handler):
        self._base.removeHandler(handler)

    def _prepare_msg(self, msg, level, indent_level):
        if indent_level is None:
            indent_level = self.current_indent
        return _format_logging_message(msg, level, indent_level, self.indent_space_num)

    def debug(self, msg, indent_level: int = None, *args, **kwargs):
        self._base.debug(self._prepare_msg(msg, logging.DEBUG, indent_level), *args, **kwargs)

    def info(self, msg, indent_level: int = None, *args, **kwargs):
        self._base.info(self._prepare_msg(msg, logging.INFO, indent_level), *args, **kwargs)

    def warning(self, msg, indent_level: int = None, *args, **kwargs):
        self._base.warning(self._prepare_msg(msg, logging.WARNING, indent_level), *args, **kwargs)

    def error(self, msg, indent_level: int = None, *args, **kwargs):
        self._base.error(self._prepare_msg(msg, logging.ERROR, indent_level), *args, **kwargs)

    def exception(self, msg, indent_level: int = None, *args, **kwargs):
        self._base.exception(self._prepare_msg(msg, logging.ERROR, indent_level), *args, **kwargs)

    def critical(self, msg, indent_level: int = None, *args, **kwargs):
        self._base.critical(self._prepare_msg(msg, logging.CRITICAL, indent_level), *args, **kwargs)

    # allow access to underlying logger when necessary
    @property
    def base_logger(self):
        return self._base


_base_logger = _setup_logger()
logger = CFELogger(_base_logger)


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

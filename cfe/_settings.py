from ._logging import logger
import warnings


class CellFateExplorerConfig:
    def __init__(self):
        # warning
        self.filter_warning = True

        # plot settings
        self.plot_format = "pdf"

        # backend settings
        self.backend = "python_function"  # ["python_function", "cfe_docker", "dynverse_docker"]

        # check if rpy2 is available
        try:
            import rpy2
            self.r_available = True
            logger.debug(f"R and rpy2{rpy2} is available. You can use dynverse backend.")
        except ImportError:
            self.r_available = False
            logger.warning("R not available. You cannot use dynverse backend.")
        # manually settings r_available=False for testing
        self.r_available = False

        self.sns_palette = "Set3"

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return None

    def __setitem__(self, key, value):
        setattr(self, key, value)


settings = CellFateExplorerConfig()

if settings.filter_warning:
    import warnings
    import sys
    import os

    sys.stderr = open(os.devnull, "w")

    os.environ["PYTHONWARNINGS"] = "ignore"

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

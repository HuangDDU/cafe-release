import os
import sys
import warnings

from ._logging import logger


class CellFateExplorerConfig:
    def __init__(self):
        # data settings
        self.data_dir = "/root/PyCode/scRNA/data"

        # backend settings
        # ["python_function", "cfe_docker", "dynverse_docker", "conda"]
        self.backend = "conda"
        self.seperate_log_file = True  # set seperate log file for each backend run

        # warning
        self.filter_warning = True

        # save external data after trajectory inference, may waste time and disk space
        self.save_external_data = False

        # plot settings
        self.plot_format = "pdf"
        self.sns_palette = "Set3"

        self.dynverse_docker_via_json = False  # middle file json result in low efficiency

        # check if rpy2 is available
        try:
            import rpy2

            self.r_available = True
            logger.debug(f"R and rpy2{rpy2} are available. You can use dynverse dataset")
        except ImportError:
            self.r_available = False
            logger.warning("R or rpy2 is not available. You can't use dynverse dataset")
        # manually settings r_available=False for testing
        self.r_available = False

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return None

    def __setitem__(self, key, value):
        setattr(self, key, value)


settings = CellFateExplorerConfig()

if settings.filter_warning:
    sys.stderr = open(os.devnull, "w")

    os.environ["PYTHONWARNINGS"] = "ignore"

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

from .._logging import logger
from ..data import FateAnnData
from .fate_backend import Backend


class FunctionBackend(Backend):
    """Specific implementation of abstract Backend class using Python functions in now conda environment."""

    def __init__(self, function_name="comp1"):
        """Initialize the FunctionBackend class.

        Args:
            function_name (str, optional):  name of the function backend .
        """
        logger.debug("FunctionBackend __init__")

        self.function_name = function_name
        self.load_backend()

    def load_backend(self) -> None:
        """load backend from python function"""
        self._load_function(self.function_name)

    def run(self, fadata: FateAnnData, parameters: dict) -> None:
        """call the python function to get trajectory dict

        Args:
            fadata (FateAnnData): the input FateAnnData object for trajectory inference method
            parameters (dict):  the  parameters for trajectory inference method
        """
        parameters = self._get_parameters(fadata, parameters)

        adata = fadata.to_anndata(delete_trajectory=True)  # avoid other trajectory IO

        trajectory_dict = self.function(adata, **parameters)

        fadata.add_trajectory_by_type(trajectory_dict)

    # TOOD: consider if __call__ is needed
    # def __call__(self, adata: AnnData, rewrite: bool = True, **parameters):
    #     """simplified version for self.run"""
    #     # transfer FateAnndata to AnnData to avoid other trajectory IO
    #     trajectory_dict = self.function(adata, **parameters)
    #     return trajectory_dict

    def __str__(self):
        return f"FunctionBackend:'{self.function_name}'"

    def install_pipy_package(self):
        # TODO: install the relevant package from pipy
        logger.debug("install_pipy_package")

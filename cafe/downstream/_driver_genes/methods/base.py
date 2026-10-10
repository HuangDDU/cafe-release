from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ....data import FateAnnData
from ..._lineage import Lineage
from ..result import DriverGeneResult


@dataclass(frozen=True)
class DriverContext:
    """Shared data and lineage inputs, without method-specific options."""

    fadata: FateAnnData
    lineage: Lineage
    selected_lineages: tuple[str, ...]

    @property
    def model_name(self) -> str:
        """Derive the model from the prepared lineage instead of duplicating it."""
        return self.lineage.model_name


class BaseDriverMethod(ABC):
    """Method protocol; core calls validate before compute on the same instance."""

    name: str

    @abstractmethod
    def validate(self, context: DriverContext, **kwargs: Any) -> None:
        """Check method inputs and dependencies without fitting the model."""

    @abstractmethod
    def compute(self, context: DriverContext, **kwargs: Any) -> DriverGeneResult:
        """Compute using inputs already validated by core."""

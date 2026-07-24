from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ...data import FateAnnData
from .._lineage import Lineage
from .._result import DriverGeneResult


@dataclass(frozen=True)
class DriverContext:
    """Validated common inputs supplied to a driver-gene method."""

    fadata: FateAnnData
    lineage: Lineage
    selected_lineages: tuple[str, ...]
    group_scope: str
    model_name: str


# Base class for various driver-gene algorithms.
class BaseDriverMethod(ABC):
    """Internal protocol implemented by driver-gene methods."""

    name: str

    @abstractmethod
    def validate(self, context: DriverContext, **kwargs: Any) -> None:
        """Validate method-specific inputs."""

    @abstractmethod
    def compute(self, context: DriverContext, **kwargs: Any) -> DriverGeneResult:
        """Compute and return a standardized result."""


_METHODS: dict[str, type[BaseDriverMethod]] = {}


def register_driver_method(name: str):
    """Register an internal driver-gene method class."""
    if not isinstance(name, str) or not name:
        raise ValueError("Driver-gene method name must be a non-empty string.")

    def decorator(method_class: type[BaseDriverMethod]):
        if name in _METHODS:
            raise ValueError(f"Driver-gene method {name!r} is already registered.")
        if not issubclass(method_class, BaseDriverMethod):
            raise TypeError("Registered driver-gene methods must inherit BaseDriverMethod.")
        _METHODS[name] = method_class
        return method_class

    return decorator


def get_driver_method(name: str) -> BaseDriverMethod:
    """Return a new instance of a registered method."""
    try:
        method_class = _METHODS[name]
    except KeyError as error:
        raise KeyError(f"Unknown driver-gene method {name!r}. Available methods: {sorted(_METHODS)!r}.") from error
    return method_class()

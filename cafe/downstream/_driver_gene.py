from collections.abc import Sequence
from typing import Any, Hashable

from ..data import FateAnnData
from ._drivers import DriverContext, get_driver_method
from ._lineage import Lineage, extract_lineages
from ._result import DriverGeneResult


class DriverGene:
    """Prepare lineages and run downstream driver-gene methods."""

    def __init__(self, fadata: FateAnnData, model_name: str | None = None):
        if not isinstance(fadata, FateAnnData):
            raise TypeError(f"Expected FateAnnData, found {type(fadata).__name__!r}.")
        parsed_model_name = fadata.parse_model_name(model_name)
        if parsed_model_name is None:
            raise KeyError(f"Trajectory model {model_name!r} is not available.")

        self.fadata = fadata
        self.model_name = parsed_model_name
        self.lineages: Lineage | None = None
        self.results: dict[str, DriverGeneResult] = {}

    def compute_lineages(
        self,
        *,
        start_milestone: Hashable | None = None,
        start_cell: str | None = None,
        terminal_states: Sequence[Hashable] | None = None,
    ) -> Lineage:
        """Extract and cache terminal lineages for the selected trajectory."""
        self.lineages = extract_lineages(
            self.fadata,
            model_name=self.model_name,
            start_milestone=start_milestone,
            start_cell=start_cell,
            terminal_states=terminal_states,
        )
        return self.lineages

    def compute(
        self,
        method: str,
        *,
        lineages: str | Sequence[str] | None = None,
        result_key: str | None = None,
        group_scope: str = "exclusive",
        **kwargs: Any,
    ) -> DriverGeneResult:
        """Run a registered driver-gene method on prepared lineages."""
        if self.lineages is None:
            raise RuntimeError("Compute lineages first with compute_lineages().")

        if lineages is None:
            selected = tuple(self.lineages.names)
        elif isinstance(lineages, str):
            selected = (lineages,)
        else:
            selected = tuple(lineages)

        unknown = [name for name in selected if name not in self.lineages.names]
        if unknown:
            raise KeyError(f"Unknown lineages: {unknown!r}.")

        context = DriverContext(
            fadata=self.fadata,
            lineage=self.lineages,
            selected_lineages=selected,
            group_scope=group_scope,
            model_name=self.model_name,
        )
        result = get_driver_method(method).compute(context, **kwargs)
        self.results[result_key or method] = result
        return result

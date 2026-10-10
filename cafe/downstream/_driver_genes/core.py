from collections.abc import Sequence
from typing import Any, Hashable

from ...data import FateAnnData
from .._lineage import extract_lineages
from .methods import DriverContext, get_method_class
from .result import DriverGeneResult


def driver_genes(
    fadata: FateAnnData,
    method: str = "de",
    *,
    model_name: str | None = None,
    lineages: str | Sequence[str] | None = None,
    start_milestone: Hashable | None = None,
    start_cell: str | None = None,
    group_scope: str = "exclusive",
    key_added: str | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> DriverGeneResult:
    """Prepare terminal lineages and run a driver-gene method in one call.

    Args:
        fadata: Expression data with a rooted directed acyclic trajectory.
        method: Supported method name. Currently "de" is available.
        model_name: Trajectory model; defaults to the active model.
        lineages: Terminal lineages included in the analysis; defaults to all.
        start_milestone: Root milestone. If omitted, use stored prior information.
        start_cell: Root cell, used when no root milestone is available.
        group_scope: Cell assignment policy; DE requires "exclusive".
        key_added: Optional key for saving the successful result in FateAnnData.
        overwrite: Allow replacing a saved result with the same key.
        **kwargs: Method options, such as DE's test, reference, layer and min_cells.

    Returns:
        Result with the complete table and a top_genes query helper.

    Notes:
        Expression data are not normalized automatically. DE uses X unless
        a layer or use_raw=True is provided. Shared trunk cells are excluded
        from DE. Without key_added, no downstream result is persisted.
    """
    method_class = get_method_class(method)  # Resolve once before preparing lineages.
    if key_added is not None:
        if not isinstance(key_added, str) or not key_added.strip():
            raise ValueError("key_added must be a non-empty string.")
        store = fadata.cafe_dict.get("downstream", {}).get("driver_gene", {})
        if key_added in store and not overwrite:
            raise KeyError(f"Driver-gene result {key_added!r} already exists.")

    lineage = extract_lineages(
        fadata,
        model_name=model_name,
        start_milestone=start_milestone,
        start_cell=start_cell,
    )
    if lineages is None:
        selected = tuple(lineage.names)
    elif isinstance(lineages, str):
        selected = (lineages,)
    else:
        selected = tuple(lineages)

    if len(set(selected)) != len(selected):
        raise ValueError("Selected lineages must be unique.")
    unknown = [name for name in selected if name not in lineage.names]
    if unknown:
        raise KeyError(f"Unknown lineages: {unknown!r}.")

    context = DriverContext(fadata=fadata, lineage=lineage, selected_lineages=selected)
    algorithm = method_class()
    algorithm.validate(context, group_scope=group_scope, **kwargs)
    result = algorithm.compute(context, group_scope=group_scope, **kwargs)
    if key_added is not None:
        result.write_to_fadata(fadata, key_added, overwrite=overwrite)
    return result

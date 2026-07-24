from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..data import FateAnnData


@dataclass
class DriverGeneResult:
    """Standard result returned by downstream driver-gene methods."""

    table: pd.DataFrame
    method: str
    params: dict[str, Any]
    lineages: tuple[str, ...]
    model_name: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to values supported by AnnData serialization."""
        return {
            "table": self.table.copy(),
            "method": self.method,
            "params": dict(self.params),
            "lineages": list(self.lineages),
            "model_name": self.model_name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DriverGeneResult":
        """Reconstruct a result from its serialized representation."""
        return cls(
            table=value["table"].copy(),
            method=str(value["method"]),
            params=dict(value["params"]),
            lineages=tuple(value["lineages"]),
            model_name=str(value["model_name"]),
            metadata=dict(value["metadata"]),
        )

    def write_to_fadata(self, fadata: FateAnnData, key: str, *, overwrite: bool = False) -> None:
        """Persist this result under Cafe's downstream namespace."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Result key must be a non-empty string.")

        store = fadata.cafe_dict.setdefault("downstream", {}).setdefault("driver_gene", {})
        if key in store and not overwrite:
            raise KeyError(f"Driver-gene result {key!r} already exists.")
        store[key] = self.to_dict()

    @classmethod
    def from_fadata(cls, fadata: FateAnnData, key: str) -> "DriverGeneResult":
        """Load a persisted result from a FateAnnData object."""
        store = fadata.cafe_dict.get("downstream", {}).get("driver_gene", {})
        if key not in store:
            raise KeyError(f"Driver-gene result {key!r} was not found.")
        return cls.from_dict(store[key])

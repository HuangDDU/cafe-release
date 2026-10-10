from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import pandas as pd

from ...data import FateAnnData


@dataclass
class DriverGeneResult:
    """Standard result returned by downstream driver-gene methods."""

    table: pd.DataFrame
    method: str
    params: dict[str, Any]
    lineages: tuple[str, ...]
    model_name: str
    metadata: dict[str, Any]

    def top_genes(self, n: int = 5, *, qval: float | None = None, direction: str = "both") -> pd.DataFrame:
        """Return up to n DE candidates per lineage without changing the table.

        qval is an optional inclusive adjusted-p-value threshold. The up and
        down directions select positive and negative logfoldchange.
        Scores are sorted descending for both/up and ascending for down.
        Thus both preserves signed-score ranking, not absolute significance
        ranking. Ties preserve the original row order.
        This query currently supports DE results only.
        """
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be a positive integer.")
        if qval is not None and (isinstance(qval, bool) or not isinstance(qval, Real) or not 0 <= qval <= 1):
            raise ValueError("qval must be a finite number between 0 and 1.")
        if direction not in ("both", "up", "down"):
            raise ValueError("direction must be 'both', 'up', or 'down'.")
        if self.method != "de":
            raise ValueError("top_genes currently supports DE results only.")

        required = ["gene", "lineage", "score"]
        if qval is not None:
            required.append("qval")
        if direction != "both":
            required.append("logfoldchange")
        missing = [column for column in required if column not in self.table]
        if missing:
            raise ValueError(f"Result table is missing required statistics: {missing!r}.")

        table = self.table
        if qval is not None:
            table = table.loc[table["qval"] <= qval]
        if direction == "up":
            table = table.loc[table["logfoldchange"] > 0]
        elif direction == "down":
            table = table.loc[table["logfoldchange"] < 0]
        if table.empty:
            return table.copy()
        ranked = [
            group.sort_values("score", ascending=direction == "down", kind="stable", na_position="last").head(n)
            for _, group in table.groupby("lineage", sort=False, observed=True)
        ]
        return pd.concat(ranked).copy()

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

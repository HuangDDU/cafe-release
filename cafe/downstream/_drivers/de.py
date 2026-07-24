from typing import Any

import pandas as pd
import scanpy as sc

from ..._settings import settings
from .._result import DriverGeneResult
from ._base import BaseDriverMethod, DriverContext, register_driver_method


@register_driver_method("de")
class DEMethod(BaseDriverMethod):
    """Terminal-lineage differential expression using Scanpy."""

    name = "de"

    def validate(
        self,
        context: DriverContext,
        *,
        layer: str | None = None,
        use_raw: bool = False,
        min_cells: int = 3,
        **_: Any,
    ) -> None:
        if len(context.selected_lineages) < 2:
            raise ValueError("Differential expression requires at least 2 lineages.")
        if context.group_scope != "exclusive":
            raise ValueError("Differential expression currently requires group_scope='exclusive'.")
        if min_cells < 1:
            raise ValueError("min_cells must be a positive integer.")
        if layer is not None and use_raw:
            raise ValueError("A named layer cannot be combined with use_raw=True.")
        if layer is not None and layer not in context.fadata.layers:
            raise KeyError(f"Expression layer {layer!r} was not found.")
        if use_raw and context.fadata.raw is None:
            raise ValueError("use_raw=True requires fadata.raw.")

        unknown = [name for name in context.selected_lineages if name not in context.lineage.names]
        if unknown:
            raise KeyError(f"Unknown lineages: {unknown!r}.")

        membership = context.lineage.exclusive_membership.loc[:, list(context.selected_lineages)]
        counts = membership.sum(axis=0)
        too_small = counts[counts < min_cells]
        if not too_small.empty:
            details = ", ".join(f"{name}={int(count)}" for name, count in too_small.items())
            raise ValueError(f"Each lineage requires at least {min_cells} cells; found {details}.")

    def compute(
        self,
        context: DriverContext,
        *,
        test: str = "wilcoxon",
        reference: str = "rest",
        layer: str | None = None,
        use_raw: bool = False,
        min_cells: int = 3,
        **kwargs: Any,
    ) -> DriverGeneResult:
        self.validate(
            context,
            layer=layer,
            use_raw=use_raw,
            min_cells=min_cells,
        )

        selected = context.selected_lineages
        membership = context.lineage.exclusive_membership.loc[:, list(selected)]
        assigned = membership.sum(axis=1).eq(1)
        labels = membership.idxmax(axis=1).where(assigned)

        adata = context.fadata[assigned].copy()
        adata.obs["_cafe_driver_lineage"] = pd.Categorical(
            labels.loc[assigned],
            categories=list(selected),
        )
        sc.tl.rank_genes_groups(
            adata,
            groupby="_cafe_driver_lineage",
            groups=list(selected),
            reference=reference,
            method=test,
            layer=layer,
            use_raw=use_raw,
            **kwargs,
        )

        tables = []
        for lineage_name in selected:
            table = sc.get.rank_genes_groups_df(adata, group=lineage_name).rename(
                columns={
                    "names": "gene",
                    "scores": "score",
                    "logfoldchanges": "logfoldchange",
                    "pvals": "pval",
                    "pvals_adj": "qval",
                }
            )
            if reference == "rest" and len(selected) == 2:
                reference_name = next(name for name in selected if name != lineage_name)
            else:
                reference_name = reference
            table["lineage"] = lineage_name
            table["reference"] = reference_name
            table["method"] = self.name
            table["comparison"] = f"{lineage_name} vs {reference_name}"
            tables.append(table)

        required = [
            "gene",
            "lineage",
            "reference",
            "score",
            "logfoldchange",
            "pval",
            "qval",
            "method",
            "comparison",
        ]
        result_table = pd.concat(tables, ignore_index=True)
        result_table = result_table[required + [column for column in result_table if column not in required]]
        cell_counts = membership.sum(axis=0).astype(int).to_dict()
        params = {
            "test": test,
            "reference": reference,
            "layer": layer,
            "use_raw": use_raw,
            "min_cells": min_cells,
            "group_scope": context.group_scope,
            **kwargs,
        }
        return DriverGeneResult(
            table=result_table,
            method=self.name,
            params=params,
            lineages=selected,
            model_name=context.model_name,
            metadata={
                "schema_version": 1,
                "expression_source": "raw" if use_raw else (layer or "X"),
                "cell_counts": cell_counts,
                "cafe_version": settings.version,
            },
        )

import numpy as np
import pandas as pd
import pytest

import cafe
from cafe.downstream._drivers import DriverContext, get_driver_method
from cafe.downstream._lineage import extract_lineages


def make_de_fadata():
    shared = ["root", "trunk"]
    alpha = [f"alpha_{i}" for i in range(4)]
    beta = [f"beta_{i}" for i in range(4)]
    obs_names = shared + alpha + beta

    expression = np.ones((len(obs_names), 3), dtype=float)
    expression[[obs_names.index(cell) for cell in alpha], 0] = 12.0
    expression[[obs_names.index(cell) for cell in beta], 1] = 12.0
    fadata = cafe.data.FateAnnData(
        X=expression,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=["Gcg", "Ins1", "Housekeeping"]),
    )
    fadata.add_model_name("tree")
    fadata.add_trajectory(
        milestone_network=pd.DataFrame(
            {
                "from": ["Root", "Branch", "Branch"],
                "to": ["Branch", "Alpha", "Beta"],
                "length": [1.0, 1.0, 1.0],
                "directed": [True, True, True],
            }
        ),
        progressions=pd.DataFrame(
            {
                "cell_id": obs_names,
                "from": ["Root", "Root"] + ["Branch"] * 4 + ["Branch"] * 4,
                "to": ["Root", "Branch"] + ["Alpha"] * 4 + ["Beta"] * 4,
                "percentage": [1.0, 0.5] + [0.5] * 8,
            }
        ),
        generate_color=False,
    )
    return fadata


def make_context(fadata):
    lineages = extract_lineages(
        fadata,
        model_name="tree",
        start_milestone="Root",
        terminal_states=["Alpha", "Beta"],
    )
    return DriverContext(
        fadata=fadata,
        lineage=lineages,
        selected_lineages=("Alpha", "Beta"),
        group_scope="exclusive",
        model_name="tree",
    )


def test_de_method_is_registered():
    assert get_driver_method("de").name == "de"

    with pytest.raises(KeyError, match="Unknown driver-gene method"):
        get_driver_method("missing")


def test_de_finds_terminal_lineage_markers_without_mutating_fadata():
    fadata = make_de_fadata()
    context = make_context(fadata)
    original_obs_columns = fadata.obs.columns.copy()

    result = get_driver_method("de").compute(context, test="wilcoxon")

    assert set(result.table.columns) >= {
        "gene",
        "lineage",
        "reference",
        "score",
        "logfoldchange",
        "pval",
        "qval",
        "method",
        "comparison",
    }
    alpha = result.table.query("lineage == 'Alpha'").sort_values("score", ascending=False)
    beta = result.table.query("lineage == 'Beta'").sort_values("score", ascending=False)
    assert alpha.iloc[0]["gene"] == "Gcg"
    assert beta.iloc[0]["gene"] == "Ins1"
    assert result.metadata["cell_counts"] == {"Alpha": 4, "Beta": 4}
    assert fadata.obs.columns.equals(original_obs_columns)
    assert "rank_genes_groups" not in fadata.uns


def test_de_validates_group_size_and_expression_source():
    fadata = make_de_fadata()
    context = make_context(fadata)
    method = get_driver_method("de")

    with pytest.raises(ValueError, match="at least 5 cells"):
        method.compute(context, min_cells=5)
    with pytest.raises(KeyError, match="layer"):
        method.compute(context, layer="missing")
    with pytest.raises(ValueError, match="use_raw"):
        method.compute(context, layer="counts", use_raw=True)

import numpy as np
import pandas as pd
import pytest

import cafe


# simulate a simple tree with two lineages, Alpha and Beta, that share a common ancestor
@pytest.fixture
def fadata():
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

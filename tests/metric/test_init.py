import pandas as pd
import pytest

import cfe


def test_init():
    metrics = cfe.metric.metrics
    assert isinstance(metrics, pd.DataFrame)
    assert set(["isomorphic", "edge_flip", "him"]) <= set(metrics["metric_id"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])

import pandas as pd
import pytest

import cafe


def test_init():
    metrics = cafe.metric.metrics
    assert isinstance(metrics, pd.DataFrame)
    assert set(["isomorphic", "edge_flip", "him"]) <= set(metrics["metric_id"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])

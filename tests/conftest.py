"""Shared deterministic data for validation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n_dates = 12
    rows_per_date = 6
    n_rows = n_dates * rows_per_date
    index = pd.Index(
        [f"ROW_{i:03d}" for i in range(n_rows)], name="ROW_ID"
    )
    signal = rng.normal(size=n_rows)
    X = pd.DataFrame(
        {
            "TS": np.repeat(
                [f"DATE_{i:02d}" for i in range(n_dates)], rows_per_date
            ),
            "ALLOCATION": np.tile(
                [f"A{i}" for i in range(rows_per_date)], n_dates
            ),
            "GROUP": np.tile(
                ["G0", "G0", "G1", "G1", "G2", "G2"], n_dates
            ),
            "signal": signal,
        },
        index=index,
    )
    target = signal + 0.25 * rng.normal(size=n_rows)
    y = pd.DataFrame(
        {
            "target": target,
            "target_binarized": (target > 0).astype(int),
        },
        index=index,
    )
    return X, y

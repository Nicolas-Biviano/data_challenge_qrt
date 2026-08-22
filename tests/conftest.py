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


@pytest.fixture
def model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ALLOCATION": ["A", "B", "A", "B", "A", "B"],
            "GROUP": ["G1", "G1", "G2", "G2", "G1", "G2"],
            "RET_1": [-0.3, 0.2, np.nan, 0.5, -0.1, 0.7],
            "RET_2": [-0.2, 0.1, 0.3, 0.4, -0.2, 0.6],
            "RET_3": [-0.1, 0.0, 0.2, 0.3, -0.3, 0.5],
            "RET_4": [-0.4, 0.3, 0.1, 0.2, -0.4, 0.4],
            "RET_7": [-0.2, 0.4, 0.0, 0.1, -0.5, 0.3],
            "RET_8": [-0.3, 0.5, -0.1, 0.0, -0.6, 0.2],
            "RET_9": [-0.4, 0.6, -0.2, -0.1, -0.7, 0.1],
            "RET_18": [-0.5, 0.7, -0.3, -0.2, -0.8, 0.0],
        }
    )

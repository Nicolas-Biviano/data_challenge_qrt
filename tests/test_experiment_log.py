from __future__ import annotations

import csv
import json

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.cross_validation import CVConfig, ModelConfig, run_cv
from src.experiment_log import log_candidate


def _classifier_pipeline():
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [("signal", StandardScaler(), ["signal"])]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=0.7,
                    solver="liblinear",
                    random_state=3,
                ),
            ),
        ]
    )


def test_log_candidate_writes_header_and_row(classification_data, tmp_path):
    X, y = classification_data
    pipeline = _classifier_pipeline()
    result = run_cv(
        X,
        y,
        ModelConfig(model=pipeline),
        CVConfig(n_splits=4, random_state=11),
    )
    log_path = tmp_path / "candidates.csv"

    log_candidate(
        "logistic_v1", result, pipeline, notes="baseline", path=log_path
    )

    with log_path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["candidate"] == "logistic_v1"
    assert row["notes"] == "baseline"
    assert row["n_folds"] == "4"
    assert float(row["accuracy"]) == pytest.approx(result.mean_accuracy)

    params = json.loads(row["params"])
    assert params["classifier__C"] == pytest.approx(0.7)


def test_log_candidate_appends_without_duplicating_header(
    classification_data, tmp_path
):
    X, y = classification_data
    pipeline = _classifier_pipeline()
    result = run_cv(
        X,
        y,
        ModelConfig(model=pipeline),
        CVConfig(n_splits=4, random_state=11),
    )
    log_path = tmp_path / "candidates.csv"

    log_candidate("first", result, pipeline, path=log_path)
    log_candidate("second", result, pipeline, path=log_path)

    with log_path.open() as handle:
        lines = handle.readlines()
        handle.seek(0)
        rows = list(csv.DictReader(handle))

    assert lines[0].startswith("timestamp,")
    assert [row["candidate"] for row in rows] == ["first", "second"]

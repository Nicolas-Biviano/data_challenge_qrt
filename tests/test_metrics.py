import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    build_validation_report,
    calibration_table,
    compare_oof,
    expected_calibration_error,
    grouped_accuracy_summary,
)


def _oof() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "TS": ["D1", "D1", "D2", "D2"],
            "ALLOCATION": ["A", "B", "A", "B"],
            "y_true": [-0.4, 0.3, -0.1, 0.8],
            "y_true_binarized": [0, 1, 0, 1],
            "prediction": [0, 1, 1, 1],
            "score": [0.1, 0.8, 0.7, 0.9],
            "is_correct": [1, 1, 0, 1],
        },
        index=pd.Index([10, 11, 12, 13], name="ROW_ID"),
    )


def test_validation_report_covers_stability_positive_rate_and_calibration():
    report = build_validation_report(_oof(), n_calibration_bins=4)

    assert report["classification"]["accuracy"] == pytest.approx(0.75)
    assert report["classification"]["target_positive_rate"] == 0.5
    assert report["classification"]["prediction_positive_rate"] == 0.75
    assert set(report["stability"]) == {"fold", "date", "allocation"}
    assert report["calibration"]["ece"] >= 0
    assert not report["calibration"]["table"].empty


def test_grouped_accuracy_penalty_uses_global_accuracy():
    summary = grouped_accuracy_summary(_oof(), "fold", penalty=1.0)
    expected_se = np.std([1.0, 0.5], ddof=1) / np.sqrt(2)

    assert summary["accuracy"] == 0.75
    assert summary["standard_error"] == pytest.approx(expected_se)
    assert summary["penalized_score"] == pytest.approx(0.75 - expected_se)


def test_calibration_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibration_table([0, 1], [0.2, 1.2])


def test_expected_calibration_error_uses_bin_weights():
    table = pd.DataFrame({"n": [3, 1], "absolute_gap": [0.1, 0.5]})
    assert expected_calibration_error(table) == pytest.approx(0.2)


def test_compare_oof_is_paired_by_row_and_group():
    baseline = _oof()
    candidate = baseline.copy()
    candidate.loc[12, "is_correct"] = 1

    comparison = compare_oof(candidate, baseline, group="TS")
    assert comparison["mean_accuracy_gain"] == 0.25
    assert comparison["groups_won"] == 1
    assert comparison["groups_tied"] == 1


def test_compare_oof_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="indices"):
        compare_oof(_oof().iloc[::-1], _oof())

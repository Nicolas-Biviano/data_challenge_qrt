"""Metrics and model-comparison helpers for out-of-fold predictions."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


__all__ = [
    "build_validation_report",
    "calibration_table",
    "classification_metrics",
    "compare_oof",
    "expected_calibration_error",
    "grouped_accuracy_summary",
    "regression_metrics",
]


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
) -> dict[str, Any]:
    """Compute the project's classification diagnostics.

    Parameters
    ----------
    y_true
        Binary target labels.
    y_pred
        Binary predicted labels aligned with ``y_true``.
    y_score
        Positive-class probabilities aligned with ``y_true``. Probability
        metrics are omitted when scores are not supplied.

    Returns
    -------
    dict
        Accuracy, balanced accuracy, F1, precision, recall, Matthews
        correlation, confusion matrix, and observed and predicted positive
        rates. AUC, Brier score, and log loss are also returned when
        ``y_score`` is supplied.

    Notes
    -----
    AUC is reported as ``NaN`` for a single-class sample so that an otherwise
    valid fold or smoke test can still be evaluated.
    """
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "f1": f1_score(truth, prediction, zero_division=0),
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "mcc": matthews_corrcoef(truth, prediction),
        "confusion_matrix": confusion_matrix(truth, prediction),
        "target_positive_rate": float(np.mean(truth)),
        "prediction_positive_rate": float(np.mean(prediction)),
    }
    if y_score is not None:
        score = np.asarray(y_score, dtype=float)
        metrics["auc"] = (
            roc_auc_score(truth, score) if np.unique(truth).size > 1 else np.nan
        )
        metrics["brier"] = brier_score_loss(truth, score)
        metrics["log_loss"] = log_loss(truth, score, labels=[0, 1])
    return metrics


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute the regression metrics used by legacy experiments.

    Parameters
    ----------
    y_true
        Continuous target values.
    y_pred
        Continuous predictions aligned with ``y_true``.

    Returns
    -------
    dict of str to float
        Mean squared error, mean absolute error, median absolute error, and
        coefficient of determination.
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def grouped_accuracy_summary(
    oof: pd.DataFrame,
    group: Hashable,
    *,
    penalty: float = 1.0,
) -> dict[str, float | int]:
    """Summarize OOF accuracy and its dispersion across groups.

    Parameters
    ----------
    oof
        Out-of-fold table containing ``is_correct`` and the grouping column.
    group
        Column whose distinct values define the stability units.
    penalty
        Multiplier applied to the simple standard error of group accuracies.

    Returns
    -------
    dict
        Row-weighted accuracy, unweighted mean and standard deviation of group
        accuracies, simple standard error, penalty, penalized score, and group
        count.

    Raises
    ------
    ValueError
        If required columns are missing or no non-missing groups are available.

    Notes
    -----
    The standard error treats group-level accuracies as the sampling units. It
    is a stability diagnostic rather than a multi-way clustered econometric
    standard error.
    """
    _require_columns(oof, {group, "is_correct"})
    accuracy_by_group = oof.groupby(group, observed=True)["is_correct"].mean()
    if accuracy_by_group.empty:
        raise ValueError(f"No non-missing groups found in {group!r}")
    accuracy = float(oof["is_correct"].mean())
    standard_error = float(
        accuracy_by_group.std(ddof=1) / np.sqrt(len(accuracy_by_group))
    )
    if len(accuracy_by_group) == 1:
        standard_error = 0.0
    return {
        "accuracy": accuracy,
        "group_mean_accuracy": float(accuracy_by_group.mean()),
        "group_std_accuracy": float(accuracy_by_group.std(ddof=0)),
        "standard_error": standard_error,
        "penalty": float(penalty),
        "penalized_score": accuracy - penalty * standard_error,
        "n_groups": int(len(accuracy_by_group)),
    }


def calibration_table(
    y_true: Any,
    y_probability: Any,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build an equal-width probability calibration table.

    Parameters
    ----------
    y_true
        Binary target labels.
    y_probability
        Positive-class probabilities in the closed interval ``[0, 1]``.
    n_bins
        Number of equal-width probability intervals.

    Returns
    -------
    pandas.DataFrame
        Non-empty bins with their observation count, mean probability,
        empirical positive rate, and absolute calibration gap.

    Raises
    ------
    ValueError
        If inputs have different lengths, fewer than two bins are requested,
        or probabilities are missing or outside ``[0, 1]``.
    """
    truth = pd.Series(np.asarray(y_true), name="y_true")
    probability = pd.Series(
        np.asarray(y_probability, dtype=float), name="probability"
    )
    if len(truth) != len(probability):
        raise ValueError("y_true and y_probability must have the same length")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if probability.isna().any() or not probability.between(0.0, 1.0).all():
        raise ValueError("Probabilities must be finite and between 0 and 1")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    frame = pd.concat((truth, probability), axis=1)
    frame["bin"] = pd.cut(
        frame["probability"], bins=bins, include_lowest=True, labels=False
    )
    table = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("y_true", "size"),
            mean_probability=("probability", "mean"),
            positive_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    table["absolute_gap"] = (
        table["mean_probability"] - table["positive_rate"]
    ).abs()
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Compute expected calibration error from a calibration table.

    Parameters
    ----------
    table
        Calibration table containing ``n`` and ``absolute_gap`` columns, as
        produced by :func:`calibration_table`.

    Returns
    -------
    float
        Absolute calibration gaps averaged with bin counts as weights.

    Raises
    ------
    ValueError
        If required columns are missing or the table contains no observations.
    """
    _require_columns(table, {"n", "absolute_gap"})
    total = table["n"].sum()
    if total == 0:
        raise ValueError("Calibration table is empty")
    return float((table["n"] * table["absolute_gap"]).sum() / total)


def build_validation_report(
    oof: pd.DataFrame,
    *,
    date_column: str = "TS",
    allocation_column: str = "ALLOCATION",
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Build the core OOF performance, stability, and calibration report.

    Parameters
    ----------
    oof
        Out-of-fold table containing labels, predictions, scores, correctness,
        and fold identifiers.
    date_column
        Optional date-group column used for stability diagnostics.
    allocation_column
        Optional allocation column used for stability diagnostics.
    n_calibration_bins
        Number of equal-width bins used in the calibration table.

    Returns
    -------
    dict
        Nested classification metrics, available stability summaries, and
        calibration diagnostics.

    Raises
    ------
    ValueError
        If required OOF columns are missing or calibration inputs are invalid.

    Notes
    -----
    Date and allocation stability entries are included only when their columns
    are present. Fold stability is always required.
    """
    _require_columns(
        oof,
        {
            "y_true_binarized",
            "prediction",
            "score",
            "is_correct",
            "fold",
        },
    )
    report: dict[str, Any] = {
        "classification": classification_metrics(
            oof["y_true_binarized"], oof["prediction"], oof["score"]
        ),
        "stability": {"fold": grouped_accuracy_summary(oof, "fold")},
    }
    for label, column in (("date", date_column), ("allocation", allocation_column)):
        if column in oof:
            report["stability"][label] = grouped_accuracy_summary(oof, column)
    table = calibration_table(
        oof["y_true_binarized"],
        oof["score"],
        n_bins=n_calibration_bins,
    )
    report["calibration"] = {
        "ece": expected_calibration_error(table),
        "table": table,
    }
    return report


def compare_oof(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    group: str = "TS",
) -> dict[str, float | int]:
    """Compare candidate and baseline OOF accuracy on identical observations.

    Parameters
    ----------
    candidate
        Candidate OOF table containing ``is_correct`` and ``group``.
    baseline
        Baseline OOF table with the same ordered index and group assignments.
    group
        Column defining the units used to summarize paired accuracy gains.

    Returns
    -------
    dict
        Row-weighted and group-mean accuracy gains, their simple group-level
        standard error and 95% interval, win/tie/loss counts, and group count.

    Raises
    ------
    ValueError
        If required columns are missing or the OOF indices or groups are not
        exactly aligned.

    Notes
    -----
    Gains are paired by observation before aggregation. The reported interval
    is descriptive and does not account for additional panel dependence.
    """
    _require_columns(candidate, {group, "is_correct"})
    _require_columns(baseline, {group, "is_correct"})
    if not candidate.index.equals(baseline.index):
        raise ValueError("Candidate and baseline OOF indices are not aligned")
    if not candidate[group].equals(baseline[group]):
        raise ValueError(f"Candidate and baseline {group!r} columns differ")

    row_gain = candidate["is_correct"].astype(float) - baseline[
        "is_correct"
    ].astype(float)
    group_gain = row_gain.groupby(candidate[group], observed=True).mean()
    standard_error = float(group_gain.std(ddof=1) / np.sqrt(len(group_gain)))
    if len(group_gain) == 1:
        standard_error = 0.0
    mean_gain = float(row_gain.mean())
    return {
        "mean_accuracy_gain": mean_gain,
        "group_mean_gain": float(group_gain.mean()),
        "group_gain_standard_error": standard_error,
        "ci95_low": mean_gain - 1.96 * standard_error,
        "ci95_high": mean_gain + 1.96 * standard_error,
        "groups_won": int((group_gain > 0).sum()),
        "groups_tied": int((group_gain == 0).sum()),
        "groups_lost": int((group_gain < 0).sum()),
        "n_groups": int(len(group_gain)),
    }


def _require_columns(frame: pd.DataFrame, required: set[Hashable]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(map(str, missing))}")

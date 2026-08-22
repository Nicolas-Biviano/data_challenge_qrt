"""Append-only comparison trail of tracked candidate models."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cross_validation import CVResult
from .metrics import build_validation_report


__all__ = ["log_candidate"]

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = ROOT / "research" / "candidates.csv"
_FIELDNAMES = [
    "timestamp",
    "candidate",
    "notes",
    "params",
    "n_folds",
    "accuracy",
    "balanced_accuracy",
    "auc",
    "f1",
    "fold_stderr",
]


def log_candidate(
    name: str,
    cv_result: CVResult,
    estimator: Any,
    *,
    notes: str = "",
    path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append one candidate's headline metrics to the tracked-research log.

    This is the tier-ii step of the research workflow described in
    ``CONTRIBUTING.md``: a candidate worth remembering, sitting between
    disposable scratch experiments and a promoted, narrated presentation
    notebook.

    Parameters
    ----------
    name
        Candidate label under which this run is recorded.
    cv_result
        Validation result produced by :func:`src.cross_validation.run_cv`.
    estimator
        Estimator or pipeline whose ``get_params()`` output is stored
        alongside the metrics, so a logged row stays reproducible.
    notes
        Free-text rationale for keeping or rejecting the candidate.
    path
        Destination CSV. Created with a header row if it does not exist yet.

    Notes
    -----
    Only a deliberately narrow set of metrics is kept: accuracy, balanced
    accuracy, AUC, and F1 from
    :func:`src.metrics.build_validation_report`, plus the fold-level
    accuracy standard error from :meth:`CVResult.score_summary`. That
    standard error is a model-selection stability diagnostic, not an
    econometric standard error under panel dependence. This log is a
    cross-candidate comparison trail, not a substitute for the full
    validation report.
    """
    classification = build_validation_report(cv_result.oof_results)["classification"]
    stability = cv_result.score_summary(grouper="fold")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate": name,
        "notes": notes,
        "params": json.dumps(estimator.get_params(), default=str, sort_keys=True),
        "n_folds": cv_result.n_folds,
        "accuracy": classification["accuracy"],
        "balanced_accuracy": classification["balanced_accuracy"],
        "auc": classification["auc"],
        "f1": classification["f1"],
        "fold_stderr": stability["standard_error"],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

"""Legacy-compatible cross-validation API.

The public ``run_cv`` function is retained for the existing research scripts.
New code can pass a complete scikit-learn ``Pipeline`` as ``ModelConfig.model``;
the estimator is cloned and fitted independently in every date-grouped fold.
The separate preprocessing callback remains temporarily available for backward
compatibility and will be removed after the model migration.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from .metrics import (
    classification_metrics,
    grouped_accuracy_summary,
    regression_metrics,
)
from .utils import Cols


logger = logging.getLogger(__name__)
GroupColumn = Literal["TS", "fold", "ALLOCATION"]
Preprocessor = Callable[[Any, Any], tuple[Any, Any]]


@dataclass
class ModelConfig:
    """Configuration for a legacy CV experiment.

    Prefer putting preprocessing inside ``model`` as a scikit-learn Pipeline.
    ``preprocessing`` exists only for the current research scripts.
    """

    model: Any
    regression: bool = False
    features: list[str] | None = None
    preprocessing: Preprocessor | None = None
    threshold_classification: float = 0.5
    threshold_regression: float = 0.0


@dataclass
class CVConfig:
    """Date-grouped cross-validation configuration."""

    n_splits: int = 8
    shuffle: bool = True
    random_state: int | None = 0
    verbose: bool = False


@dataclass
class FoldResult:
    """Model, predictions and metrics for one external fold."""

    fold_id: int
    train_metrics: dict[str, Any]
    valid_metrics: dict[str, Any]
    train_index: pd.Series
    valid_index: pd.Series
    train_dates: np.ndarray
    valid_dates: np.ndarray
    fitted_model: Any = None
    y_score_train: np.ndarray | None = None
    y_score_test: np.ndarray | None = None
    y_binary_train: np.ndarray | None = None
    y_binary_test: np.ndarray | None = None


@dataclass
class CVResult:
    """Out-of-fold predictions and per-fold results."""

    n_folds: int
    oof_results: pd.DataFrame
    fold_results: list[FoldResult]
    mean_accuracy: float = np.nan

    def get_fold_results(self, i: int) -> FoldResult:
        return self.fold_results[i]

    def score(
        self,
        penalty: float = 1.0,
        grouper: GroupColumn = "fold",
    ) -> float:
        """Return global accuracy minus grouped standard-error penalty."""

        summary = grouped_accuracy_summary(
            self.oof_results, grouper, penalty=penalty
        )
        return float(summary["penalized_score"])

    def score_summary(
        self,
        penalty: float = 1.0,
        grouper: GroupColumn = "fold",
    ) -> dict[str, float | int]:
        """Return all components of the penalized score."""

        return grouped_accuracy_summary(
            self.oof_results, grouper, penalty=penalty
        )


# Compatibility aliases used throughout the historical scripts.
CVconfig = CVConfig
CVresults = CVResult
FoldResults = FoldResult


def run_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_config: ModelConfig,
    cv_config: CVConfig | None = None,
) -> CVResult:
    """Run shuffled K-fold validation while keeping every date intact.

    The function preserves the historical API and OOF column names.  The
    estimator (which may be a complete sklearn Pipeline) is cloned per fold.
    """

    cv_config = CVConfig() if cv_config is None else cv_config
    _validate_inputs(X, y, cv_config)
    if cv_config.verbose:
        logger.setLevel(logging.INFO)

    date_col = Cols.DATE.value
    unique_dates = X[date_col].unique()
    splitter = KFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=cv_config.random_state if cv_config.shuffle else None,
    )
    oof = _empty_oof(y.index)
    fold_results: list[FoldResult] = []
    target = Cols.TARGET.value
    binary_target = Cols.TARGET_BIN.value

    for fold_id, (train_positions, valid_positions) in enumerate(
        splitter.split(unique_dates), start=1
    ):
        logger.info("Processing fold %s/%s", fold_id, cv_config.n_splits)
        train_dates = unique_dates[train_positions]
        valid_dates = unique_dates[valid_positions]
        _validate_fold_dates(train_dates, valid_dates)

        train_mask = X[date_col].isin(train_dates)
        valid_mask = X[date_col].isin(valid_dates)
        X_train = X.loc[train_mask]
        X_valid = X.loc[valid_mask]
        y_train = y.loc[train_mask]
        y_valid = y.loc[valid_mask]

        X_train_ready, X_valid_ready = _preprocess_fold_data(
            X_train, X_valid, model_config
        )
        model, train_score, valid_score = _fit_model_and_predict(
            X_train_ready, X_valid_ready, y_train, model_config
        )

        if model_config.regression:
            train_prediction = (
                train_score > model_config.threshold_regression
            ).astype(int)
            valid_prediction = (
                valid_score > model_config.threshold_regression
            ).astype(int)
            train_metrics = regression_metrics(
                y_train[target], train_score
            ) | classification_metrics(
                y_train[binary_target], train_prediction
            )
            valid_metrics = regression_metrics(
                y_valid[target], valid_score
            ) | classification_metrics(
                y_valid[binary_target], valid_prediction
            )
        else:
            train_prediction = (
                train_score > model_config.threshold_classification
            ).astype(int)
            valid_prediction = (
                valid_score > model_config.threshold_classification
            ).astype(int)
            train_metrics = classification_metrics(
                y_train[binary_target], train_prediction, train_score
            )
            valid_metrics = classification_metrics(
                y_valid[binary_target], valid_prediction, valid_score
            )

        fold_results.append(
            FoldResult(
                fold_id=fold_id,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                train_index=train_mask,
                valid_index=valid_mask,
                train_dates=train_dates,
                valid_dates=valid_dates,
                fitted_model=model,
                y_score_train=train_score,
                y_score_test=valid_score,
                y_binary_train=train_prediction,
                y_binary_test=valid_prediction,
            )
        )
        oof.loc[valid_mask, "fold"] = fold_id
        oof.loc[valid_mask, "TS"] = X_valid[date_col].to_numpy()
        if Cols.ALLOCATION.value in X_valid:
            oof.loc[valid_mask, "ALLOCATION"] = X_valid[
                Cols.ALLOCATION.value
            ].to_numpy()
        oof.loc[valid_mask, "y_true"] = y_valid[target].to_numpy()
        oof.loc[valid_mask, "y_true_binarized"] = y_valid[
            binary_target
        ].to_numpy()
        oof.loc[valid_mask, "score"] = np.asarray(valid_score)
        oof.loc[valid_mask, "prediction"] = np.asarray(valid_prediction)
        logger.info("Fold accuracy=%.4f", valid_metrics["accuracy"])

    _validate_oof_coverage(oof, expected_folds=cv_config.n_splits)
    oof["is_correct"] = (
        oof["prediction"] == oof["y_true_binarized"]
    ).astype(int)
    return CVResult(
        n_folds=cv_config.n_splits,
        oof_results=oof,
        fold_results=fold_results,
        mean_accuracy=float(oof["is_correct"].mean()),
    )


def _empty_oof(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": np.nan,
            "TS": None,
            "ALLOCATION": None,
            "y_true": np.nan,
            "y_true_binarized": np.nan,
            "score": np.nan,
            "prediction": np.nan,
            "is_correct": np.nan,
        },
        index=index,
    )


def _validate_inputs(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv_config: CVConfig,
) -> None:
    if not X.index.equals(y.index):
        raise ValueError("X and y do not share the same ordered index")
    required_targets = {Cols.TARGET.value, Cols.TARGET_BIN.value}
    missing_targets = required_targets - set(y.columns)
    if missing_targets:
        raise ValueError(f"Missing target columns: {sorted(missing_targets)}")
    if Cols.DATE.value not in X:
        raise ValueError(f"X has no {Cols.DATE.value!r} date column")
    if X.index.has_duplicates or y.index.has_duplicates:
        raise ValueError("X and y indices must be unique")
    if X[Cols.DATE.value].isna().any():
        raise ValueError("Date groups cannot contain missing values")
    if y[list(required_targets)].isna().any().any():
        raise ValueError("Targets cannot contain missing values")
    n_dates = X[Cols.DATE.value].nunique()
    if not 2 <= cv_config.n_splits <= n_dates:
        raise ValueError(
            f"n_splits must be between 2 and the number of dates ({n_dates})"
        )


def _validate_fold_dates(
    train_dates: np.ndarray,
    valid_dates: np.ndarray,
) -> None:
    overlap = np.intersect1d(train_dates, valid_dates)
    if overlap.size:
        raise RuntimeError(f"Date leakage detected: {overlap.tolist()}")


def _validate_oof_coverage(oof: pd.DataFrame, expected_folds: int) -> None:
    required = [
        "fold",
        "TS",
        "y_true",
        "y_true_binarized",
        "score",
        "prediction",
    ]
    missing_by_column = oof[required].isna().sum()
    missing = missing_by_column[missing_by_column.gt(0)]
    if not missing.empty:
        raise RuntimeError(
            f"Incomplete OOF coverage: {missing.to_dict()} missing values"
        )
    observed_folds = set(oof["fold"].astype(int).unique())
    required_folds = set(range(1, expected_folds + 1))
    if observed_folds != required_folds:
        raise RuntimeError(
            f"Unexpected OOF folds: observed={sorted(observed_folds)}, "
            f"expected={sorted(required_folds)}"
        )


# Historical helper names kept for direct imports in notebooks.
def _check_data(X: pd.DataFrame, y: pd.DataFrame) -> None:
    _validate_inputs(X, y, CVConfig(n_splits=2))


def _preprocess_fold_data(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    model_config: ModelConfig,
) -> tuple[Any, Any]:
    if model_config.features is not None:
        X_train = X_train[model_config.features]
        X_valid = X_valid[model_config.features]
    else:
        drop_columns = [
            column
            for column in (
                Cols.ALLOCATION.value,
                Cols.DATE.value,
                Cols.GROUP.value,
            )
            if column in X_train
        ]
        X_train = X_train.drop(columns=drop_columns)
        X_valid = X_valid.drop(columns=drop_columns)
    if model_config.preprocessing is not None:
        X_train, X_valid = model_config.preprocessing(X_train, X_valid)
    return X_train, X_valid


def _fit_model_and_predict(
    X_train: Any,
    X_valid: Any,
    y_train: pd.DataFrame,
    model_config: ModelConfig,
) -> tuple[Any, np.ndarray, np.ndarray]:
    model = clone(model_config.model)
    if model_config.regression:
        model.fit(X_train, y_train[Cols.TARGET.value])
        return model, np.asarray(model.predict(X_train)), np.asarray(
            model.predict(X_valid)
        )

    model.fit(X_train, y_train[Cols.TARGET_BIN.value])
    return model, _positive_class_probability(model, X_train), _positive_class_probability(
        model, X_valid
    )


def _positive_class_probability(model: Any, X: Any) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Classification estimators must implement predict_proba; "
            "wrap preprocessing and the estimator in an sklearn Pipeline."
        )
    probabilities = np.asarray(model.predict_proba(X))
    classes = np.asarray(model.classes_)
    positions = np.flatnonzero(classes == 1)
    if positions.size != 1:
        raise ValueError("Classification estimator has no unique positive class 1")
    return probabilities[:, positions[0]]


def _score_model_regression(y_true: Any, y_pred: Any) -> dict[str, float]:
    return regression_metrics(y_true, y_pred)


def _score_model_classification(
    y_true: Any,
    y_pred_binary: Any,
    y_pred_score: Any | None = None,
) -> dict[str, Any]:
    return classification_metrics(y_true, y_pred_binary, y_pred_score)

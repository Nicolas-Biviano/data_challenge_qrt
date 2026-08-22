"""Canonical end-to-end modelling recipes."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.pipeline import Pipeline

from .models import (
    FIRST_RIDGE_ALPHA,
    RETAINED_LOGISTIC_C,
    make_logistic_classifier,
    make_ridge_regressor,
    make_lgbm
)
from .preprocessing import CATEGORICAL_FEATURES, make_fixed_effect_preprocessor


__all__ = [
    "FIRST_MODEL_NUMERIC_FEATURES",
    "make_first_ridge_pipeline",
    "make_retained_pipeline",
]

FIRST_MODEL_NUMERIC_FEATURES = (
    "RET_1",
    "RET_2",
    "RET_3",
    "RET_4",
    "RET_7",
    "RET_8",
    "RET_9",
    "RET_18",
)
"""Numeric columns selected for the first ridge model.

Chosen to exclude noise from the full 20-lag return history: recent lags
(1-4) for short-term signal, a weekly cluster (7-9), and a longer lag (18,
roughly one trading month) for trend.
"""


def make_retained_pipeline(
    *,
    C: float = RETAINED_LOGISTIC_C,
    random_state: int = 0,
    interaction_scale: float = 1.0,
    numeric_features: Sequence[str] = FIRST_MODEL_NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    """Build the retained sparse logistic-classification pipeline.

    Parameters
    ----------
    C
        Inverse L2 regularization strength.
    random_state
        Random seed passed to logistic regression.
    interaction_scale
        Multiplicative scale applied to allocation-specific ``RET_1`` slopes.
    numeric_features
        Numeric return columns fed to the preprocessor.
    categorical_features
        Categorical columns fed to the preprocessor.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted feature, preprocessing, and classification recipe.

    Notes
    -----
    Defaults to ``FIRST_MODEL_NUMERIC_FEATURES`` — this pipeline currently
    reuses the first model's column selection unmodified rather than an
    independently re-derived one.
    """
    return Pipeline(
        [
            (
                "preprocessor",
                make_fixed_effect_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    interaction_scale=interaction_scale,
                ),
            ),
            (
                "classifier",
                make_logistic_classifier(C=C, random_state=random_state),
            ),
        ]
    )


def make_first_ridge_pipeline(
    *,
    alpha: float = FIRST_RIDGE_ALPHA,
    interaction_scale: float = 1.0,
    numeric_features: Sequence[str] = FIRST_MODEL_NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    """Build the first internal continuous-target Ridge pipeline.

    Parameters
    ----------
    alpha
        L2 regularization strength.
    interaction_scale
        Multiplicative scale applied to allocation-specific ``RET_1`` slopes.
    numeric_features
        Numeric return columns fed to the preprocessor.
    categorical_features
        Categorical columns fed to the preprocessor.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted feature, preprocessing, and regression recipe.
    """
    return Pipeline(
        [
            (
                "preprocessor",
                make_fixed_effect_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    interaction_scale=interaction_scale,
                ),
            ),
            ("regressor", make_ridge_regressor(alpha=alpha)),
        ]
    )

def make_lgbm_pipe_v1() -> Pipeline:
    return Pipeline(
        [
            (
                "preprocessor",
                make_fixed_effect_preprocessor(
                    interaction_scale=interaction_scale
                ),
            ),
            ("regressor", make_ridge_regressor(alpha=alpha)),
        ]
    )

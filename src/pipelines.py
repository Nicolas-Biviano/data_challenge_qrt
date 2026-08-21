"""Canonical end-to-end modelling recipes."""

from sklearn.pipeline import Pipeline

from .models import (
    FIRST_RIDGE_ALPHA,
    RETAINED_LOGISTIC_C,
    make_logistic_classifier,
    make_ridge_regressor,
    make_lgbm
)
from .preprocessing import make_fixed_effect_preprocessor


__all__ = ["make_first_ridge_pipeline", "make_retained_pipeline"]


def make_retained_pipeline(
    *,
    C: float = RETAINED_LOGISTIC_C,
    random_state: int = 0,
    interaction_scale: float = 1.0,
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

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted feature, preprocessing, and classification recipe.
    """
    return Pipeline(
        [
            (
                "preprocessor",
                make_fixed_effect_preprocessor(
                    interaction_scale=interaction_scale
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
) -> Pipeline:
    """Build the first internal continuous-target Ridge pipeline.

    Parameters
    ----------
    alpha
        L2 regularization strength.
    interaction_scale
        Multiplicative scale applied to allocation-specific ``RET_1`` slopes.

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
                    interaction_scale=interaction_scale
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

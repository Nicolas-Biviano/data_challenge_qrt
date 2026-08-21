"""Estimator factories without feature or validation logic."""

from sklearn.linear_model import LogisticRegression, Ridge
from lightgbm import LGBMRegressor


RETAINED_LOGISTIC_C = 0.003
FIRST_RIDGE_ALPHA = 100.0

__all__ = [
    "FIRST_RIDGE_ALPHA",
    "RETAINED_LOGISTIC_C",
    "make_logistic_classifier",
    "make_ridge_regressor",
]


def make_logistic_classifier(
    *,
    C: float = RETAINED_LOGISTIC_C,
    random_state: int = 0,
) -> LogisticRegression:
    """Build the retained L2-regularized logistic estimator.

    Parameters
    ----------
    C
        Inverse L2 regularization strength.
    random_state
        Random seed passed to the estimator.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Unfitted binary classifier.
    """
    return LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=150,
        tol=1e-5,
        random_state=random_state,
    )


def make_ridge_regressor(
    *,
    alpha: float = FIRST_RIDGE_ALPHA,
) -> Ridge:
    """Build the first internal Ridge estimator.

    Parameters
    ----------
    alpha
        L2 regularization strength.

    Returns
    -------
    sklearn.linear_model.Ridge
        Unfitted continuous-target estimator.
    """
    return Ridge(alpha=alpha, solver="lsqr")

def make_lgbm() -> LGBMRegressor:
    """Build a lgbm regressor.

    Parameters
    ----------

    Returns
    -------
    lightgbm.LGBMRegressor
        Unfitted continuous-target estimator.
    """    
    return LGBMRegressor(
        boosting_type    = "gbdt",
        max_depth        = 3, 
        num_leaves       = 2 ** 3,
        learning_rate    = 1e-3,
        n_estimators     = 1_000,
        min_split_gain   = 0.01,
        subsample        = 0.7, 
        colsample_bytree = 0.7,
        n_jobs           = -1,
        reg_lambda       = 100,
        reg_alpha        = 1,
    )

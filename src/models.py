"""Canonical scikit-learn preprocessing and model definitions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


NUMERIC_FEATURES = (
    "RET_1",
    "RET_2",
    "RET_3",
    "RET_4",
    "RET_7",
    "RET_8",
    "RET_9",
    "RET_18",
)
CATEGORICAL_FEATURES = ("ALLOCATION", "GROUP")
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
BASELINE_C = 0.003

__all__ = [
    "BASELINE_C",
    "CATEGORICAL_FEATURES",
    "MODEL_FEATURES",
    "NUMERIC_FEATURES",
    "AllocationReturnInteraction",
    "make_baseline_v2",
    "make_fixed_effect_preprocessor",
]


class AllocationReturnInteraction(BaseEstimator, TransformerMixin):
    """Create allocation-specific slopes for one return variable.

    Parameters
    ----------
    allocation_column
        Categorical column identifying allocations.
    return_column
        Numeric return column interacted with the allocation indicators.
    interaction_scale
        Multiplicative scale applied to the resulting interaction block.

    Notes
    -----
    The return is median-imputed and standardized using the training fold only.
    Unknown allocations produce an all-zero interaction row at transform time.
    This is the only project-specific transformer in the retained pipeline.
    """

    def __init__(
        self,
        allocation_column: str = "ALLOCATION",
        return_column: str = "RET_1",
        interaction_scale: float = 1.0,
    ) -> None:
        """Initialize the unfitted interaction transformer."""
        self.allocation_column = allocation_column
        self.return_column = return_column
        self.interaction_scale = interaction_scale

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "AllocationReturnInteraction":
        """Learn imputation, scaling, and allocation levels.

        Parameters
        ----------
        X
            Frame containing the allocation and return columns.
        y
            Ignored target, accepted for scikit-learn compatibility.

        Returns
        -------
        AllocationReturnInteraction
            Fitted transformer.
        """
        frame = self._validate_frame(X)
        self.return_pipeline_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.allocation_encoder_ = OneHotEncoder(handle_unknown="ignore")
        self.return_pipeline_.fit(frame[[self.return_column]])
        self.allocation_encoder_.fit(frame[[self.allocation_column]])
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        """Transform observations into sparse allocation-return interactions.

        Parameters
        ----------
        X
            Frame containing the allocation and return columns.

        Returns
        -------
        scipy.sparse.csr_matrix
            One column per allocation level observed during fitting.
        """
        check_is_fitted(self, ("return_pipeline_", "allocation_encoder_"))
        frame = self._validate_frame(X)
        scaled_return = self.return_pipeline_.transform(
            frame[[self.return_column]]
        )
        allocation = self.allocation_encoder_.transform(
            frame[[self.allocation_column]]
        )
        interactions = allocation.multiply(scaled_return[:, [0]])
        interactions *= self.interaction_scale
        return interactions.tocsr()

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        """Return output names for the fitted interaction columns.

        Parameters
        ----------
        input_features
            Ignored input names, accepted for scikit-learn compatibility.

        Returns
        -------
        numpy.ndarray
            Names combining each encoded allocation with the return column.
        """
        check_is_fitted(self, "allocation_encoder_")
        allocation_names = self.allocation_encoder_.get_feature_names_out(
            [self.allocation_column]
        )
        return np.asarray(
            [f"{name} x {self.return_column}" for name in allocation_names],
            dtype=object,
        )

    def _validate_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "AllocationReturnInteraction expects a pandas DataFrame"
            )
        required = {self.allocation_column, self.return_column}
        missing = required - set(X.columns)
        if missing:
            raise ValueError(f"Missing interaction features: {sorted(missing)}")
        return X.loc[:, [self.allocation_column, self.return_column]]


def make_fixed_effect_preprocessor(
    *,
    interaction_scale: float = 1.0,
) -> ColumnTransformer:
    """Build the retained model's complete preprocessing graph.

    Parameters
    ----------
    interaction_scale
        Multiplicative scale applied to allocation-specific ``RET_1`` slopes.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Cloneable sparse preprocessor with numeric returns, categorical fixed
        effects, and allocation-specific ``RET_1`` slopes.

    Notes
    -----
    The standard scikit-learn blocks remain visible through
    ``named_transformers_`` after fitting. Every learned parameter is therefore
    fitted inside the enclosing validation-fold pipeline.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("returns", numeric_pipeline, list(NUMERIC_FEATURES)),
            (
                "fixed_effects",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "allocation_ret1",
                AllocationReturnInteraction(
                    interaction_scale=interaction_scale
                ),
                ["ALLOCATION", "RET_1"],
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def make_baseline_v2(
    *,
    C: float = BASELINE_C,
    random_state: int = 0,
) -> Pipeline:
    """Build the retained sparse logistic-classification pipeline.

    Parameters
    ----------
    C
        Inverse L2 regularization strength passed to logistic regression.
    random_state
        Random seed passed to logistic regression.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted preprocessing and logistic-classification pipeline.

    Notes
    -----
    Passing this complete object to date-grouped validation ensures that
    imputation, scaling, encoding, and estimation are all learned within each
    training fold.
    """
    return Pipeline(
        [
            ("preprocessor", make_fixed_effect_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=150,
                    tol=1e-5,
                    random_state=random_state,
                ),
            ),
        ]
    )

"""Interaction transformers for heterogeneous allocation effects."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


__all__ = ["AllocationReturnInteraction"]


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

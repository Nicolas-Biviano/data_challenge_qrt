"""Canonical model definitions for the QRT challenge."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
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


class FixedEffectDesign(BaseEstimator, TransformerMixin):
    """Build the sparse design used by the retained V2 classifier.

    Every learned preprocessing step lives in this sklearn transformer, so a
    complete Pipeline can be cloned and fitted independently inside each fold.
    """

    def __init__(self, interaction_scale: float = 1.0):
        self.interaction_scale = interaction_scale

    def fit(self, X: pd.DataFrame, y=None) -> "FixedEffectDesign":
        frame = self._select_and_validate(X)
        self.numeric_pipeline_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.category_encoder_ = OneHotEncoder(handle_unknown="ignore")
        self.allocation_encoder_ = OneHotEncoder(handle_unknown="ignore")

        self.numeric_pipeline_.fit(frame[list(NUMERIC_FEATURES)])
        self.category_encoder_.fit(frame[list(CATEGORICAL_FEATURES)])
        self.allocation_encoder_.fit(frame[["ALLOCATION"]])
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(MODEL_FEATURES, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        check_is_fitted(
            self,
            ("numeric_pipeline_", "category_encoder_", "allocation_encoder_"),
        )
        frame = self._select_and_validate(X)
        numeric = self.numeric_pipeline_.transform(frame[list(NUMERIC_FEATURES)])
        categories = self.category_encoder_.transform(
            frame[list(CATEGORICAL_FEATURES)]
        )
        allocation = self.allocation_encoder_.transform(frame[["ALLOCATION"]])
        allocation_ret1 = allocation.multiply(numeric[:, 0, None])
        allocation_ret1 *= self.interaction_scale
        return sparse.hstack(
            (sparse.csr_matrix(numeric), categories, allocation_ret1),
            format="csr",
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, ("category_encoder_", "allocation_encoder_"))
        category_names = self.category_encoder_.get_feature_names_out(
            CATEGORICAL_FEATURES
        )
        allocation_names = self.allocation_encoder_.get_feature_names_out(
            ["ALLOCATION"]
        )
        interaction_names = np.asarray(
            [f"{name} x RET_1" for name in allocation_names], dtype=object
        )
        return np.concatenate(
            (
                np.asarray(NUMERIC_FEATURES, dtype=object),
                category_names,
                interaction_names,
            )
        )

    def get_feature_names(self) -> list[str]:
        """Backward-compatible list form used by the research scripts."""

        return self.get_feature_names_out().tolist()

    @staticmethod
    def _select_and_validate(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FixedEffectDesign expects a pandas DataFrame")
        missing = set(MODEL_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        return X.loc[:, MODEL_FEATURES]


def make_baseline_v2(
    *,
    C: float = BASELINE_C,
    random_state: int = 0,
) -> Pipeline:
    """Return the complete retained V2 preprocessing-and-model Pipeline."""

    return Pipeline(
        [
            ("design", FixedEffectDesign()),
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

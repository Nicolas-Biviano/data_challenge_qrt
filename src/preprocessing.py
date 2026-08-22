"""Preprocessing graphs that assemble reusable feature blocks."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import AllocationReturnInteraction


CATEGORICAL_FEATURES = ("ALLOCATION", "GROUP")

__all__ = [
    "CATEGORICAL_FEATURES",
    "make_fixed_effect_preprocessor",
]


def make_fixed_effect_preprocessor(
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    interaction_scale: float = 1.0,
) -> ColumnTransformer:
    """Build a fixed-effect preprocessing graph for a given column selection.

    Which columns to use is a modelling decision that belongs to the calling
    pipeline (see e.g. ``src.pipelines.FIRST_MODEL_NUMERIC_FEATURES``), not a
    default owned by this module — this function only knows how to build a
    leak-safe preprocessor for whatever selection it is given.

    Parameters
    ----------
    numeric_features
        Numeric return columns to impute and standardize.
    categorical_features
        Categorical columns to one-hot encode.
    interaction_scale
        Multiplicative scale applied to allocation-specific ``RET_1`` slopes.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Cloneable sparse preprocessor with numeric returns, categorical fixed
        effects, and allocation-specific ``RET_1`` slopes.

    Notes
    -----
    Column selection lives inside this graph. It can therefore receive the raw
    challenge frame while every learned parameter is fitted inside the
    enclosing validation-fold pipeline.

    The allocation-return interaction block always uses ``ALLOCATION`` and
    ``RET_1`` specifically, regardless of ``numeric_features`` /
    ``categorical_features`` — both must be included in their respective
    argument for that block to receive its inputs.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("returns", numeric_pipeline, list(numeric_features)),
            (
                "fixed_effects",
                OneHotEncoder(handle_unknown="ignore"),
                list(categorical_features),
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

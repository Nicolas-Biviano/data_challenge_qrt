"""Preprocessing graphs that assemble reusable feature blocks."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import AllocationReturnInteraction


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

__all__ = [
    "CATEGORICAL_FEATURES",
    "MODEL_FEATURES",
    "NUMERIC_FEATURES",
    "make_fixed_effect_preprocessor",
]


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
    Column selection lives inside this graph. It can therefore receive the raw
    challenge frame while every learned parameter is fitted inside the
    enclosing validation-fold pipeline.
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

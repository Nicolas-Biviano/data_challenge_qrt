import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

from src.feature_engineering import AllocationReturnInteraction
from src.preprocessing import make_fixed_effect_preprocessor

NUMERIC_COLUMNS = ("RET_1", "RET_2", "RET_3", "RET_4", "RET_7", "RET_8", "RET_9", "RET_18")
CATEGORICAL_COLUMNS = ("ALLOCATION", "GROUP")


def test_fixed_effect_preprocessor_is_cloneable_and_sparse(model_frame):
    preprocessor = clone(
        make_fixed_effect_preprocessor(
            numeric_features=NUMERIC_COLUMNS,
            categorical_features=CATEGORICAL_COLUMNS,
        )
    ).fit(model_frame)
    matrix = preprocessor.transform(model_frame)

    assert isinstance(preprocessor, ColumnTransformer)
    assert matrix.shape[0] == len(model_frame)
    assert matrix.shape[1] == len(preprocessor.get_feature_names_out())
    assert np.isfinite(matrix.data).all()


def test_preprocessor_rejects_missing_features(model_frame):
    with pytest.raises(ValueError, match="column"):
        make_fixed_effect_preprocessor(
            numeric_features=NUMERIC_COLUMNS,
            categorical_features=CATEGORICAL_COLUMNS,
        ).fit(model_frame.drop(columns="RET_18"))

    interaction = AllocationReturnInteraction().fit(model_frame)
    unknown = model_frame.iloc[[0]].assign(ALLOCATION="unseen")
    assert interaction.transform(unknown).nnz == 0

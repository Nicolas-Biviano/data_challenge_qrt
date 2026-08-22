import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

from src.feature_engineering import AllocationReturnInteraction
from src.preprocessing import MODEL_FEATURES, make_fixed_effect_preprocessor


def test_fixed_effect_preprocessor_is_cloneable_and_sparse(model_frame):
    preprocessor = clone(make_fixed_effect_preprocessor()).fit(model_frame)
    matrix = preprocessor.transform(model_frame)

    assert isinstance(preprocessor, ColumnTransformer)
    assert matrix.shape[0] == len(model_frame)
    assert matrix.shape[1] == len(preprocessor.get_feature_names_out())
    assert np.isfinite(matrix.data).all()


def test_preprocessor_rejects_missing_features(model_frame):
    with pytest.raises(ValueError, match="column"):
        make_fixed_effect_preprocessor().fit(model_frame.drop(columns="RET_18"))

    interaction = AllocationReturnInteraction().fit(model_frame)
    unknown = model_frame.iloc[[0]].assign(ALLOCATION="unseen")
    assert interaction.transform(unknown).nnz == 0


def test_model_feature_order_is_explicit():
    assert MODEL_FEATURES[:2] == ("ALLOCATION", "GROUP")
    assert MODEL_FEATURES[2] == "RET_1"

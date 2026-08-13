import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.models import (
    MODEL_FEATURES,
    FixedEffectDesign,
    make_baseline_v2,
)


@pytest.fixture
def model_frame():
    return pd.DataFrame(
        {
            "ALLOCATION": ["A", "B", "A", "B", "A", "B"],
            "GROUP": ["G1", "G1", "G2", "G2", "G1", "G2"],
            "RET_1": [-0.3, 0.2, np.nan, 0.5, -0.1, 0.7],
            "RET_2": [-0.2, 0.1, 0.3, 0.4, -0.2, 0.6],
            "RET_3": [-0.1, 0.0, 0.2, 0.3, -0.3, 0.5],
            "RET_4": [-0.4, 0.3, 0.1, 0.2, -0.4, 0.4],
            "RET_7": [-0.2, 0.4, 0.0, 0.1, -0.5, 0.3],
            "RET_8": [-0.3, 0.5, -0.1, 0.0, -0.6, 0.2],
            "RET_9": [-0.4, 0.6, -0.2, -0.1, -0.7, 0.1],
            "RET_18": [-0.5, 0.7, -0.3, -0.2, -0.8, 0.0],
        }
    )


def test_fixed_effect_design_is_cloneable_and_sparse(model_frame):
    design = clone(FixedEffectDesign()).fit(model_frame)
    matrix = design.transform(model_frame)

    assert matrix.shape[0] == len(model_frame)
    assert matrix.shape[1] == len(design.get_feature_names_out())
    assert np.isfinite(matrix.data).all()


def test_baseline_pipeline_fits_raw_frame_and_predicts_probabilities(model_frame):
    target = np.array([0, 1, 0, 1, 0, 1])
    pipeline = clone(make_baseline_v2()).fit(model_frame, target)
    probabilities = pipeline.predict_proba(model_frame)[:, 1]

    assert list(pipeline.named_steps) == ["design", "classifier"]
    assert probabilities.shape == (len(model_frame),)
    assert np.logical_and(probabilities >= 0, probabilities <= 1).all()


def test_design_rejects_missing_features(model_frame):
    with pytest.raises(ValueError, match="RET_18"):
        FixedEffectDesign().fit(model_frame.drop(columns="RET_18"))


def test_model_feature_order_is_explicit():
    assert MODEL_FEATURES[:2] == ("ALLOCATION", "GROUP")
    assert MODEL_FEATURES[2] == "RET_1"

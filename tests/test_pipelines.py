import numpy as np
from sklearn.base import clone

from src.pipelines import FIRST_MODEL_NUMERIC_FEATURES, make_retained_pipeline
from src.preprocessing import CATEGORICAL_FEATURES


def test_retained_pipeline_fits_raw_frame_and_predicts_probabilities(model_frame):
    target = np.array([0, 1, 0, 1, 0, 1])
    pipeline = clone(make_retained_pipeline()).fit(model_frame, target)
    probabilities = pipeline.predict_proba(model_frame)[:, 1]

    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    assert probabilities.shape == (len(model_frame),)
    assert np.logical_and(probabilities >= 0, probabilities <= 1).all()


def test_retained_pipeline_default_column_selection_is_explicit():
    preprocessor = make_retained_pipeline().named_steps["preprocessor"]
    columns_by_block = {
        name: columns for name, _, columns in preprocessor.transformers
    }

    assert columns_by_block["returns"] == list(FIRST_MODEL_NUMERIC_FEATURES)
    assert columns_by_block["fixed_effects"] == list(CATEGORICAL_FEATURES)

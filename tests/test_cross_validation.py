from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.cross_validation import CVConfig, CVconfig, ModelConfig, run_cv


def _classifier_pipeline():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.7, solver="liblinear", random_state=3),
    )


def test_legacy_reference_is_unchanged(classification_data):
    X, y = classification_data
    result = run_cv(
        X,
        y,
        ModelConfig(model=_classifier_pipeline(), features=["signal"]),
        CVconfig(n_splits=4, random_state=11),
    )

    assert result.mean_accuracy == pytest.approx(0.8611111111111112)
    assert [fold.valid_metrics["accuracy"] for fold in result.fold_results] == (
        pytest.approx([0.777777777778, 1.0, 0.777777777778, 0.888888888889])
    )
    assert result.oof_results["fold"].astype(int).tolist()[:18] == [
        *([4] * 12),
        *([2] * 6),
    ]


def test_dates_are_disjoint_and_oof_is_complete(classification_data):
    X, y = classification_data
    result = run_cv(
        X,
        y,
        ModelConfig(model=_classifier_pipeline(), features=["signal"]),
        CVConfig(n_splits=4, random_state=11),
    )

    assert result.oof_results.index.equals(X.index)
    assert not result.oof_results.isna().any().any()
    assert set(result.oof_results["fold"].astype(int)) == {1, 2, 3, 4}
    for fold in result.fold_results:
        assert set(fold.train_dates).isdisjoint(fold.valid_dates)
        assert not (fold.train_index & fold.valid_index).any()
        assert (fold.train_index | fold.valid_index).all()


def test_pipeline_preprocessing_is_fitted_inside_each_fold(classification_data):
    X, y = classification_data
    result = run_cv(
        X,
        y,
        ModelConfig(model=_classifier_pipeline(), features=["signal"]),
        CVConfig(n_splits=4, random_state=11),
    )

    for fold in result.fold_results:
        fitted_scaler = fold.fitted_model.named_steps["standardscaler"]
        expected_train_mean = X.loc[fold.train_index, "signal"].mean()
        assert fitted_scaler.mean_[0] == pytest.approx(expected_train_mean)


def test_legacy_preprocessing_callback_remains_available(classification_data):
    X, y = classification_data
    fitted_means = []

    def legacy_preprocessing(X_train, X_valid):
        mean = X_train["signal"].mean()
        fitted_means.append(mean)
        return X_train - mean, X_valid - mean

    result = run_cv(
        X,
        y,
        ModelConfig(
            model=LogisticRegression(
                C=0.7, solver="liblinear", random_state=3
            ),
            features=["signal"],
            preprocessing=legacy_preprocessing,
        ),
        CVConfig(n_splits=4, random_state=11),
    )

    assert len(fitted_means) == 4
    for fitted_mean, fold in zip(fitted_means, result.fold_results):
        assert fitted_mean == pytest.approx(
            X.loc[fold.train_index, "signal"].mean()
        )


def test_same_seed_produces_identical_oof(classification_data):
    X, y = classification_data
    config = CVConfig(n_splits=4, random_state=11)
    model = ModelConfig(model=_classifier_pipeline(), features=["signal"])

    first = run_cv(X, y, model, config).oof_results
    second = run_cv(X, y, model, config).oof_results
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda X, y: (X, y.iloc[::-1]), "ordered index"),
        (
            lambda X, y: (X.drop(columns="TS"), y),
            "date column",
        ),
    ],
)
def test_invalid_inputs_fail_explicitly(classification_data, mutation, message):
    X, y = mutation(*classification_data)
    with pytest.raises(ValueError, match=message):
        run_cv(
            X,
            y,
            ModelConfig(model=_classifier_pipeline(), features=["signal"]),
            CVConfig(n_splits=4),
        )


def test_score_has_a_structured_non_printing_variant(
    classification_data, capsys
):
    X, y = classification_data
    result = run_cv(
        X,
        y,
        ModelConfig(model=_classifier_pipeline(), features=["signal"]),
        CVConfig(n_splits=4, random_state=11),
    )

    score = result.score(grouper="fold")
    summary = result.score_summary(grouper="fold")
    assert score == pytest.approx(summary["penalized_score"])
    assert capsys.readouterr().out == ""

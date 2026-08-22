from sklearn.linear_model import LogisticRegression, Ridge

from src.models import (
    FIRST_RIDGE_ALPHA,
    RETAINED_LOGISTIC_C,
    make_logistic_classifier,
    make_ridge_regressor,
)


def test_make_logistic_classifier_uses_retained_defaults():
    classifier = make_logistic_classifier()

    assert isinstance(classifier, LogisticRegression)
    assert classifier.C == RETAINED_LOGISTIC_C
    assert classifier.penalty == "l2"


def test_make_logistic_classifier_accepts_overrides():
    classifier = make_logistic_classifier(C=0.5, random_state=3)

    assert classifier.C == 0.5
    assert classifier.random_state == 3


def test_make_ridge_regressor_uses_first_ridge_default():
    regressor = make_ridge_regressor()

    assert isinstance(regressor, Ridge)
    assert regressor.alpha == FIRST_RIDGE_ALPHA


def test_make_ridge_regressor_accepts_override():
    regressor = make_ridge_regressor(alpha=10.0)

    assert regressor.alpha == 10.0

import pandas as pd
import pytest

from src.data import ChallengeDataLoader


@pytest.fixture
def loader(tmp_path, monkeypatch):
    x_train = pd.DataFrame({"ROW_ID": ["R1", "R2"], "RET_1": [0.1, -0.2]})
    y_train = pd.DataFrame({"ROW_ID": ["R1", "R2"], "target": [0.05, -0.01]})
    x_test = pd.DataFrame({"ROW_ID": ["R3"], "RET_1": [0.3]})
    sample_submission = pd.DataFrame({"ROW_ID": ["R3"], "target": [0.0]})

    x_train.to_csv(tmp_path / "X_train.csv", index=False)
    y_train.to_csv(tmp_path / "y_train.csv", index=False)
    x_test.to_csv(tmp_path / "X_test.csv", index=False)
    sample_submission.to_csv(tmp_path / "sample_submission.csv", index=False)

    monkeypatch.setattr(ChallengeDataLoader, "PATH_X_TRAIN", tmp_path / "X_train.csv")
    monkeypatch.setattr(ChallengeDataLoader, "PATH_Y_TRAIN", tmp_path / "y_train.csv")
    monkeypatch.setattr(ChallengeDataLoader, "PATH_X_TEST", tmp_path / "X_test.csv")
    monkeypatch.setattr(
        ChallengeDataLoader,
        "PATH_SAMPLE_SUBMISSION",
        tmp_path / "sample_submission.csv",
    )
    return ChallengeDataLoader


def test_load_x_train_is_indexed_by_row_id(loader):
    X = loader.load_X_train()
    assert X.index.name == "ROW_ID"
    assert list(X.index) == ["R1", "R2"]


def test_load_y_train_adds_binarized_target(loader):
    y = loader.load_y_train()
    assert list(y["target_binarized"]) == [1, 0]


def test_load_train_df_joins_predictors_and_targets(loader):
    train = loader.load_train_df()
    assert list(train.columns) == ["RET_1", "target", "target_binarized"]
    assert train.index.equals(pd.Index(["R1", "R2"], name="ROW_ID"))


def test_load_x_test_and_sample_submission_are_indexed_by_row_id(loader):
    assert list(loader.load_X_test().index) == ["R3"]
    assert list(loader.load_sample_submission().index) == ["R3"]

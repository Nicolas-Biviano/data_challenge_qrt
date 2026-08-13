"""Data access helpers for the QRT challenge files."""

from pathlib import Path

import pandas as pd


PATH_DATA = Path(__file__).resolve().parent.parent / "data"


class ChallengeDataLoader:
    """Load the challenge CSV files from the repository data directory.

    All returned frames use ``ROW_ID`` as their index. File paths are class
    attributes so tests or downstream applications can point the loader to an
    alternative data location without changing the loading logic.

    Attributes
    ----------
    PATH_X_TRAIN
        Path to the training predictors.
    PATH_Y_TRAIN
        Path to the continuous training target.
    PATH_X_TEST
        Path to the test predictors.
    PATH_SAMPLE_SUBMISSION
        Path to the sample submission file.
    """

    PATH_X_TRAIN = PATH_DATA / "X_train.csv"
    PATH_Y_TRAIN = PATH_DATA / "y_train.csv"
    PATH_X_TEST = PATH_DATA / "X_test.csv"
    PATH_SAMPLE_SUBMISSION = PATH_DATA / "sample_submission.csv"

    @classmethod
    def load_X_train(cls) -> pd.DataFrame:
        """Load the training predictors.

        Returns
        -------
        pandas.DataFrame
            Training predictors indexed by ``ROW_ID``.
        """

        return pd.read_csv(cls.PATH_X_TRAIN, index_col="ROW_ID")

    @classmethod
    def load_X_test(cls) -> pd.DataFrame:
        """Load the test predictors.

        Returns
        -------
        pandas.DataFrame
            Test predictors indexed by ``ROW_ID``.
        """

        return pd.read_csv(cls.PATH_X_TEST, index_col="ROW_ID")

    @classmethod
    def load_y_train(cls) -> pd.DataFrame:
        """Load the continuous and binarized training targets.

        Returns
        -------
        pandas.DataFrame
            Target data indexed by ``ROW_ID``. The original ``target`` column
            is accompanied by ``target_binarized``, which equals one when the
            continuous target is strictly positive.
        """

        y_train = pd.read_csv(cls.PATH_Y_TRAIN, index_col="ROW_ID")
        return y_train.assign(target_binarized=(y_train > 0).astype(int))

    @classmethod
    def load_train_df(cls) -> pd.DataFrame:
        """Load predictors and targets in one index-aligned training frame.

        Returns
        -------
        pandas.DataFrame
            Training predictors, continuous target, and binarized target
            joined on ``ROW_ID``.
        """

        return pd.concat((cls.load_X_train(), cls.load_y_train()), axis=1)

    @classmethod
    def load_sample_submission(cls) -> pd.DataFrame:
        """Load the sample submission template.

        Returns
        -------
        pandas.DataFrame
            Submission template indexed by ``ROW_ID``.
        """

        return pd.read_csv(cls.PATH_SAMPLE_SUBMISSION, index_col="ROW_ID")

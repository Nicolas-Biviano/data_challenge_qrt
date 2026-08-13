"""Column names used by the QRT challenge dataset."""

from enum import Enum


class Cols(Enum):
    """Canonical raw-data and target column names.

    Using this enumeration avoids duplicating structural column names across
    loaders, validation routines, and notebooks.
    """

    DATE = "TS"
    ALLOCATION = "ALLOCATION"
    GROUP = "GROUP"
    TARGET = "target"
    TARGET_BIN = "target_binarized"
    TURNOVER = "MEDIAN_DAILY_TURNOVER"

    @classmethod
    def ret_cols(cls, start: int = 1, end: int = 20) -> list[str]:
        """Return inclusive lagged-return column names.

        Parameters
        ----------
        start
            First lag to include.
        end
            Last lag to include.

        Returns
        -------
        list of str
            Names from ``RET_{start}`` through ``RET_{end}``.
        """

        return [f"RET_{i}" for i in range(start, end + 1)]

    @classmethod
    def vol_cols(cls, start: int = 2, end: int = 20) -> list[str]:
        """Return inclusive signed-volume lag column names.

        Parameters
        ----------
        start
            First lag to include. The default excludes ``SIGNED_VOLUME_1``,
            whose missingness is structurally different in the supplied data.
        end
            Last lag to include.

        Returns
        -------
        list of str
            Names from ``SIGNED_VOLUME_{start}`` through
            ``SIGNED_VOLUME_{end}``.
        """

        return [f"SIGNED_VOLUME_{i}" for i in range(start, end + 1)]

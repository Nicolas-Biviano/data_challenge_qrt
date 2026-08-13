"""Column names used by the QRT challenge dataset."""

from enum import Enum


class Cols(Enum):
    """Canonical raw-data and target column names."""

    DATE = "TS"
    ALLOCATION = "ALLOCATION"
    GROUP = "GROUP"
    TARGET = "target"
    TARGET_BIN = "target_binarized"
    TURNOVER = "MEDIAN_DAILY_TURNOVER"

    @classmethod
    def ret_cols(cls, start: int = 1, end: int = 20) -> list[str]:
        return [f"RET_{i}" for i in range(start, end + 1)]

    @classmethod
    def vol_cols(cls, start: int = 2, end: int = 20) -> list[str]:
        return [f"SIGNED_VOLUME_{i}" for i in range(start, end + 1)]

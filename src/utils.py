from enum import Enum

class Cols(Enum):
    DATE = "TS"
    ALLOCATION = "ALLOCATION"
    GROUP = "GROUP"
    TARGET = "target"
    TARGET_BIN = "target_binarized"
    TURNOVER = "MEDIAN_DAILY_TURNOVER"
    
    @classmethod
    def ret_cols(cls, start=1, end=20):
        return [f"RET_{i}" for i in range(start, end+1)]
    
    @classmethod
    def vol_cols(cls, start=2, end=20):
        return [f"SIGNED_VOLUME_{i}" for i in range(start, end+1)]
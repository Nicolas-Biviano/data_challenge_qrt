from src.schema import Cols


def test_column_name_values():
    assert Cols.DATE.value == "TS"
    assert Cols.ALLOCATION.value == "ALLOCATION"
    assert Cols.GROUP.value == "GROUP"
    assert Cols.TARGET.value == "target"
    assert Cols.TARGET_BIN.value == "target_binarized"
    assert Cols.TURNOVER.value == "MEDIAN_DAILY_TURNOVER"


def test_ret_cols_default_range_is_1_through_20():
    assert Cols.ret_cols() == [f"RET_{i}" for i in range(1, 21)]


def test_ret_cols_accepts_custom_range():
    assert Cols.ret_cols(3, 5) == ["RET_3", "RET_4", "RET_5"]


def test_vol_cols_default_excludes_signed_volume_1():
    cols = Cols.vol_cols()
    assert cols[0] == "SIGNED_VOLUME_2"
    assert cols[-1] == "SIGNED_VOLUME_20"


def test_vol_cols_accepts_custom_range():
    assert Cols.vol_cols(1, 3) == [
        "SIGNED_VOLUME_1",
        "SIGNED_VOLUME_2",
        "SIGNED_VOLUME_3",
    ]

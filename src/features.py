"""Feature engineering helpers for the QRT challenge dataset.

The public functions expect the raw challenge columns defined in ``Cols`` and
return DataFrames aligned on the input index.
"""

from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from .utils import Cols


FeatureBlock = Literal["market", "residual", "variance_ratio", "dispersion_herding", "interactions", "signed_volume"]
BLOCKS: tuple[FeatureBlock, ...] = ("market", "residual", "variance_ratio", "dispersion_herding", "interactions", "signed_volume")


def _row_corr(x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Compute row-wise correlations while ignoring missing values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = ~np.isnan(x) & ~np.isnan(y)
    n_valid = valid.sum(axis=1)
    xx = np.where(valid, x, 0.0)
    yy = np.where(valid, y, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        cov = n_valid * np.sum(xx * yy, axis=1) - np.sum(xx, axis=1) * np.sum(yy, axis=1)
        vx = n_valid * np.sum(xx ** 2, axis=1) - np.sum(xx, axis=1) * np.sum(xx, axis=1)
        vy = n_valid * np.sum(yy ** 2, axis=1) - np.sum(yy, axis=1) * np.sum(yy, axis=1)
        corr = cov / np.sqrt(vx * vy)

    corr[(n_valid < 3) | (vx <= 0) | (vy <= 0)] = np.nan
    return corr


def signed_volume_features(X: pd.DataFrame) -> pd.DataFrame:
    """Build signed-volume summary features.

    Returns:
        Columns ``sv_mean``, ``sv_vol`` and ``ret_sv_corr``.
    """
    ret_cols = X[Cols.ret_cols()]
    sv_cols = X[Cols.vol_cols(1, 20)]
    corr_ret_sv = pd.Series(_row_corr(ret_cols, sv_cols), name="ret_sv_corr", index=X.index)
    sv_width = sv_cols.abs().pow(0.5).add_suffix("_width")
    sv_sign = (sv_cols > 0).astype(int).add_suffix("_sign")
    sv_mean = sv_cols.mean(axis=1).rename("sv_mean")
    sv_mean = sv_cols.mean(axis=1).rename("sv_mean")
    sv_vol = sv_cols.std(axis=1).rename("sv_vol")
    return pd.concat([sv_mean, sv_width, sv_vol, sv_sign, corr_ret_sv], axis=1)


def _row_ols(y: ArrayLike, x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fit row-wise OLS models ``y_k = a + b * x_k + e_k``.

    Returns:
        Intercepts, slopes and residuals, all aligned row-wise with ``y``.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    valid = ~np.isnan(x) & ~np.isnan(y)
    n = valid.sum(axis=1)
    xx = np.where(valid, x, 0.0)
    yy = np.where(valid, y, 0.0)
    sx, sy = xx.sum(axis=1), yy.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = n * (xx * xx).sum(axis=1) - sx ** 2
        b = (n * (xx * yy).sum(axis=1) - sx * sy) / denom
        a = (sy - b * sx) / n

    bad = (n < 3) | (denom == 0)
    a[bad], b[bad] = np.nan, np.nan
    resid = np.where(valid, y - (a[:, None] + b[:, None] * x), np.nan)
    return a, b, resid


def volatility_features(X: pd.DataFrame) -> pd.DataFrame:
    """Build realized-volatility and downside semi-deviation features.

    Returns:
        Columns ``ret_vol_20``, ``downside_semidev_20`` and ``n_valid_lags``.
    """
    ret = X[Cols.ret_cols()]
    vol = ret.std(axis=1).rename("ret_vol_20")
    neg = ret.where(ret < 0)
    semidev = np.sqrt((neg ** 2).sum(axis=1) / ret.notna().sum(axis=1))
    semidev = semidev.rename("downside_semidev_20")
    # n_valid = ret.notna().sum(axis=1).rename("n_valid_lags")
    return pd.concat([vol, semidev], axis=1)


def market_path(X: pd.DataFrame, leave_one_out: bool = True) -> pd.DataFrame:
    """Compute cross-sectional market returns by date for each lag.

    Args:
        X: Input data containing ``TS`` and ``RET_1`` ... ``RET_20``.
        leave_one_out: If true, removes the current allocation from its daily
            cross-sectional mean.

    Returns:
        DataFrame aligned on ``X.index`` with columns ``MKT_1`` ... ``MKT_20``.
    """
    rc = Cols.ret_cols()
    g = X.groupby(Cols.DATE.value)[rc]
    mkt = g.transform("mean")

    if leave_one_out:
        cnt = g.transform("count")
        r = X[rc]
        with np.errstate(invalid="ignore", divide="ignore"):
            excl = (cnt * mkt - r.fillna(0.0)) / (cnt - 1)
        mkt = excl.where(r.notna(), mkt).where(cnt > 1)

    return mkt.set_axis([f"MKT_{i}" for i in range(1, 21)], axis=1)


def market_features(X: pd.DataFrame, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate the market path into return and volatility features."""
    mkt = market_path(X) if mkt is None else mkt
    return pd.concat([
        mkt.mean(axis=1).rename("mkt_ret_20"),
        mkt.std(axis=1).rename("mkt_vol_20"),
        mkt.iloc[:, 0].rename("mkt_ret_1"),
    ], axis=1)


def residual_features(X: pd.DataFrame, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build market-residual features from row-wise regressions.

    Inspired by residual momentum and idiosyncratic volatility signals from
    Blitz, Huij & Martens (2011) and Ang, Hodrick, Xing & Zhang (2006).
    """
    mkt = market_path(X) if mkt is None else mkt
    ret = X[Cols.ret_cols()]
    a, b, resid = _row_ols(ret, mkt)

    resid = pd.DataFrame(resid, index=X.index, columns=Cols.ret_cols())
    vol, idio = ret.std(axis=1), resid.std(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        share = (idio ** 2) / vol.replace(0.0, np.nan) ** 2

    return pd.concat([
        pd.Series(b, index=X.index, name="beta_20"),
        pd.Series(a, index=X.index, name="alpha_20"),
        resid[Cols.ret_cols(1, 5)].mean(axis=1).rename("resid_mom_5"),
        resid.mean(axis=1).rename("resid_mom_20"),
        idio.rename("idio_vol_20"),
        share.rename("idio_share"),
    ], axis=1)


def variance_ratio_features(X: pd.DataFrame, ks: Sequence[int] = (2, 5)) -> pd.DataFrame:
    """Build Lo & MacKinlay-style variance-ratio features.

    A ratio above 1 suggests trending behavior; below 1 suggests mean reversion.
    The same ``ddof`` is used in numerator and denominator to avoid a shifted
    ratio.
    """
    r = X[Cols.ret_cols()].to_numpy(float)
    var1 = np.nanvar(r, axis=1, ddof=1)
    n_win = r.shape[1]
    out = {}

    for k in ks:
        sums = np.stack([r[:, i:i + k].sum(axis=1) for i in range(n_win - k + 1)], axis=1)
        has_nan = np.stack([np.isnan(r[:, i:i + k]).any(axis=1) for i in range(n_win - k + 1)], axis=1)
        sums = np.where(has_nan, np.nan, sums)

        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"vr_{k}"] = np.nanvar(sums, axis=1, ddof=1) / (k * np.where(var1 > 0, var1, np.nan))

    out["ac1_20"] = _row_corr(r[:, :-1], r[:, 1:])
    return pd.DataFrame(out, index=X.index)


def dispersion_features(X: pd.DataFrame) -> pd.DataFrame:
    """Build daily cross-sectional return-dispersion features.

    Inspired by Angelidis, Sakkas & Tessaromatis (2015).
    """
    rc = Cols.ret_cols()
    disp = X.groupby(Cols.DATE.value)[rc].transform("std")
    disp_1 = disp.iloc[:, 0].rename("day_disp")
    disp_mean = disp.mean(axis=1).rename("disp_mean_20")

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (disp_1 / disp_mean.replace(0.0, np.nan)).rename("disp_now_vs_avg")

    return pd.concat([disp_1, disp_mean, ratio], axis=1)


def herding_features(X: pd.DataFrame, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build cross-sectional absolute-deviation herding features.

    Inspired by Chang, Cheng & Khorana (2000) and Christie & Huang (1995).
    """
    mkt = market_path(X, leave_one_out=False) if mkt is None else mkt
    dev = (X["RET_1"] - mkt.iloc[:, 0]).abs().rename("abs_dev_1")
    csad = dev.groupby(X[Cols.DATE.value]).transform("mean").rename("csad_1")
    abs_mkt = mkt.iloc[:, 0].abs()

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (csad / abs_mkt.replace(0.0, np.nan)).rename("csad_vs_absmkt")

    return pd.concat([dev, csad, ratio, (mkt.iloc[:, 0] ** 2).rename("mkt_1_sq")], axis=1)


def momentum_features(X: pd.DataFrame) -> pd.DataFrame:
    """Build time-series momentum and volatility-managed return features.

    Inspired by Ehsani & Linnainmaa (2022), Moskowitz, Ooi & Pedersen (2012),
    and Moreira & Muir (2017).
    """
    ret = X[Cols.ret_cols()]
    vol = ret.std(axis=1)
    out = {}

    for w in (5, 20):
        s = ret[Cols.ret_cols(1, w)].sum(axis=1, min_count=1)
        out[f"fmom_sign_{w}"] = np.sign(s)

    v = vol.replace(0.0, np.nan)
    for w in (1, 5, 20):
        out[f"tsmom_{w}"] = ret[Cols.ret_cols(1, w)].mean(axis=1) / v

    out["inv_var"] = 1.0 / (v ** 2)
    out["volmanaged_20"] = ret.mean(axis=1) / (v ** 2)
    return pd.DataFrame(out, index=X.index)


def interaction_features(
    X: pd.DataFrame,
    disp: pd.DataFrame | None = None,
    mkt_feat: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build interaction features between momentum, dispersion, volume and turnover."""
    ret = X[Cols.ret_cols()]
    mom, vol = ret.mean(axis=1), ret.std(axis=1)
    disp = dispersion_features(X) if disp is None else disp
    mkt_feat = market_features(X) if mkt_feat is None else mkt_feat
    g = X.groupby(Cols.DATE.value)
    sv_rank = g[["SIGNED_VOLUME_1"]].rank(pct=True).iloc[:, 0]
    to_rank = g[[Cols.TURNOVER.value]].rank(pct=True).iloc[:, 0]

    return pd.DataFrame({
        "mom_x_disp": mom * disp["day_disp"],
        "mom_x_vol": mom * vol,
        "mom_x_mktvol": mom * mkt_feat["mkt_vol_20"],
        "rev_x_sv": X["RET_1"] * sv_rank,
        "rev_x_turnover": X["RET_1"] * to_rank,
    }, index=X.index)


def build(X: pd.DataFrame, blocks: Sequence[str] | None = None) -> pd.DataFrame:
    """Build selected feature blocks.

    Args:
        X: Input data containing the raw challenge columns referenced by
            ``Cols``.
        blocks: Feature block names to include. Defaults to all blocks in
            ``BLOCKS``.

    Returns:
        DataFrame containing all selected features, aligned on ``X.index``.

    Raises:
        ValueError: If ``blocks`` contains an unknown block name.
    """
    if blocks is None:
        blocks = BLOCKS

    unknown_blocks = set(blocks) - set(BLOCKS)
    if unknown_blocks:
        raise ValueError(f"Unknown feature blocks: {sorted(unknown_blocks)}")

    mkt = market_path(X)
    disp = dispersion_features(X)
    mf = market_features(X, mkt)
    parts = []

    if "market" in blocks:
        parts += [mf, momentum_features(X), volatility_features(X)]
    if "residual" in blocks:
        parts += [residual_features(X, mkt)]
    if "variance_ratio" in blocks:
        parts += [variance_ratio_features(X)]
    if "dispersion_herding" in blocks:
        parts += [disp, herding_features(X)]
    if "interactions" in blocks:
        parts += [interaction_features(X, disp, mf)]
    if "signed_volume" in blocks:
        parts += [signed_volume_features(X)]

    return pd.concat(parts, axis=1)

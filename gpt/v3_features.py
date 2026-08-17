"""Features compactes et categories X-only pour les experiences V3."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_RETURNS = [
    "RET_1", "RET_2", "RET_3", "RET_4", "RET_7", "RET_8", "RET_9", "RET_18"
]
ALL_RETURNS = [f"RET_{i}" for i in range(1, 21)]
ALL_VOLUMES = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
OBSERVED_VOLUMES = [f"SIGNED_VOLUME_{i}" for i in range(2, 21)]
SELECTED_VOLUMES = [
    "SIGNED_VOLUME_1",
    "SIGNED_VOLUME_2",
    "SIGNED_VOLUME_3",
    "SIGNED_VOLUME_5",
    "SIGNED_VOLUME_10",
    "SIGNED_VOLUME_15",
    "SIGNED_VOLUME_20",
]


def _row_corr(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    x = left.to_numpy(float)
    y = right.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=1)
    xx = np.where(valid, x, 0.0)
    yy = np.where(valid, y, 0.0)
    sx, sy = xx.sum(axis=1), yy.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = n * (xx * yy).sum(axis=1) - sx * sy
        denominator = np.sqrt(
            (n * (xx * xx).sum(axis=1) - sx**2)
            * (n * (yy * yy).sum(axis=1) - sy**2)
        )
        corr = numerator / denominator
    corr[(n < 3) | ~np.isfinite(corr)] = np.nan
    return corr


def _row_slope(frame: pd.DataFrame) -> np.ndarray:
    y = frame.to_numpy(float)
    x = np.broadcast_to(np.arange(y.shape[1], dtype=float), y.shape)
    valid = np.isfinite(y)
    n = valid.sum(axis=1)
    xx = np.where(valid, x, 0.0)
    yy = np.where(valid, y, 0.0)
    sx, sy = xx.sum(axis=1), yy.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        denominator = n * (xx * xx).sum(axis=1) - sx**2
        slope = (n * (xx * yy).sum(axis=1) - sx * sy) / denominator
    slope[(n < 3) | ~np.isfinite(slope)] = np.nan
    return slope


def _sign_change_share(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(float)
    valid = np.isfinite(values[:, 1:]) & np.isfinite(values[:, :-1])
    changes = np.where(
        valid,
        np.sign(values[:, 1:]) != np.sign(values[:, :-1]),
        np.nan,
    )
    return np.nanmean(changes, axis=1)


def _add_cross_sectional(
    features: pd.DataFrame,
    columns: list[str],
    group_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    grouped = features.groupby(group_columns, observed=True)
    additions: dict[str, pd.Series] = {}
    for column in columns:
        mean = grouped[column].transform("mean")
        std = grouped[column].transform("std")
        additions[f"{prefix}_mean_{column}"] = mean.astype("float32")
        additions[f"{prefix}_std_{column}"] = std.astype("float32")
        additions[f"{prefix}_rank_{column}"] = grouped[column].rank(
            pct=True,
            method="average",
        ).astype("float32")
    return pd.concat([features, pd.DataFrame(additions, index=features.index)], axis=1)


def build_v3_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Construit les blocs returns, non-return et combined sans utiliser y."""
    features = X[["TS", "ALLOCATION", "GROUP", "MEDIAN_DAILY_TURNOVER"]].copy()
    features[BASE_RETURNS + SELECTED_VOLUMES] = X[
        BASE_RETURNS + SELECTED_VOLUMES
    ].astype("float32")
    returns = X[ALL_RETURNS]
    volumes = X[OBSERVED_VOLUMES]

    return_additions: dict[str, pd.Series | np.ndarray] = {}
    for window in (3, 5, 10, 20):
        block = returns.iloc[:, :window]
        return_additions[f"ret_mean_{window}"] = block.mean(axis=1).astype("float32")
    for window in (5, 20):
        block = returns.iloc[:, :window]
        return_additions[f"ret_std_{window}"] = block.std(axis=1).astype("float32")
        return_additions[f"ret_positive_share_{window}"] = (
            (block > 0.0).mean(axis=1).astype("float32")
        )
    return_additions["ret_slope_20"] = _row_slope(returns).astype("float32")
    return_additions["ret_ac1_20"] = _row_corr(
        returns.iloc[:, :-1], returns.iloc[:, 1:]
    ).astype("float32")
    return_additions["ret_sign_changes_20"] = _sign_change_share(returns).astype(
        "float32"
    )
    ret_std = returns.std(axis=1).replace(0.0, np.nan)
    return_additions["ret1_over_std20"] = (X["RET_1"] / ret_std).astype("float32")
    features = pd.concat(
        [features, pd.DataFrame(return_additions, index=X.index)],
        axis=1,
    )

    volume_additions: dict[str, pd.Series | np.ndarray] = {}
    volume_additions["sv_mean_5"] = volumes.iloc[:, :5].mean(axis=1).astype("float32")
    volume_additions["sv_mean_20"] = volumes.mean(axis=1).astype("float32")
    volume_additions["sv_median_20"] = volumes.median(axis=1).astype("float32")
    row_median = volumes.median(axis=1)
    volume_additions["sv_mad_20"] = volumes.sub(row_median, axis=0).abs().median(
        axis=1
    ).astype("float32")
    volume_additions["sv_std_5"] = volumes.iloc[:, :5].std(axis=1).astype("float32")
    volume_additions["sv_std_20"] = volumes.std(axis=1).astype("float32")
    volume_additions["sv_abs_mean_20"] = volumes.abs().mean(axis=1).astype("float32")
    volume_additions["sv_positive_share_20"] = (volumes > 0.0).mean(axis=1).astype(
        "float32"
    )
    volume_additions["sv_slope_20"] = _row_slope(volumes).astype("float32")
    volume_additions["sv_ac1_20"] = _row_corr(
        volumes.iloc[:, :-1], volumes.iloc[:, 1:]
    ).astype("float32")
    volume_additions["sv_sign_changes_20"] = _sign_change_share(volumes).astype(
        "float32"
    )
    volume_additions["ret_sv_corr_2_20"] = _row_corr(
        returns.iloc[:, 1:], volumes
    ).astype("float32")
    volume_additions["sv1_available"] = X["SIGNED_VOLUME_1"].notna().astype("float32")
    volume_additions["sv_missing_share_20"] = X[ALL_VOLUMES].isna().mean(axis=1).astype(
        "float32"
    )
    volume_additions["sv_recent_vs_long"] = (
        volume_additions["sv_mean_5"] - volume_additions["sv_mean_20"]
    ).astype("float32")
    volume_additions["ret1_x_turnover"] = (
        X["RET_1"] * X["MEDIAN_DAILY_TURNOVER"]
    ).astype("float32")
    features = pd.concat(
        [features, pd.DataFrame(volume_additions, index=X.index)],
        axis=1,
    )

    cross_columns = [
        "RET_1",
        "ret_mean_5",
        "SIGNED_VOLUME_1",
        "SIGNED_VOLUME_2",
        "sv_mean_20",
        "MEDIAN_DAILY_TURNOVER",
    ]
    features = _add_cross_sectional(features, cross_columns, ["TS"], "date")
    features = _add_cross_sectional(
        features,
        cross_columns,
        ["TS", "GROUP"],
        "group_date",
    )

    return_cross_columns = ["RET_1", "ret_mean_5"]
    non_return_cross_columns = [
        "SIGNED_VOLUME_1",
        "SIGNED_VOLUME_2",
        "sv_mean_20",
        "MEDIAN_DAILY_TURNOVER",
    ]
    return_cross_features = [
        f"{prefix}_{stat}_{column}"
        for prefix in ("date", "group_date")
        for column in return_cross_columns
        for stat in ("mean", "std", "rank")
    ]
    non_return_cross_features = [
        f"{prefix}_{stat}_{column}"
        for prefix in ("date", "group_date")
        for column in non_return_cross_columns
        for stat in ("mean", "std", "rank")
    ]
    return_features = BASE_RETURNS + list(return_additions) + return_cross_features
    interaction_features = ["ret1_x_turnover"]
    non_return_features = (
        ["MEDIAN_DAILY_TURNOVER"]
        + SELECTED_VOLUMES
        + [name for name in volume_additions if name not in interaction_features]
        + non_return_cross_features
    )
    blocks = {
        "returns_compact": list(dict.fromkeys(BASE_RETURNS + list(return_additions))),
        "returns": list(dict.fromkeys(return_features)),
        "non_return": list(dict.fromkeys(non_return_features)),
        "combined": list(
            dict.fromkeys(return_features + non_return_features + interaction_features)
        ),
    }
    return features, blocks


def allocation_clusters(
    X_train: pd.DataFrame,
    all_allocations: pd.Series,
    n_clusters: int = 8,
) -> pd.Series:
    """Apprend des archetypes d'allocation a partir des variables non-return."""
    profile_columns = SELECTED_VOLUMES + ["MEDIAN_DAILY_TURNOVER"]
    grouped = X_train.groupby("ALLOCATION", observed=True)[profile_columns]
    profile = grouped.agg(["mean", "std", "median"])
    profile.columns = [f"{column}_{stat}" for column, stat in profile.columns]
    missing = grouped.apply(lambda frame: frame.isna().mean(), include_groups=False)
    missing.columns = [f"{column}_missing" for column in missing.columns]
    profile = pd.concat([profile, missing], axis=1)

    n_components = min(8, profile.shape[1], len(profile) - 1)
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PCA(n_components=n_components, random_state=0),
        AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
    )
    labels = pipeline.fit_predict(profile)
    mapping = pd.Series(labels, index=profile.index)
    return all_allocations.map(mapping).fillna(-1).astype("int32")

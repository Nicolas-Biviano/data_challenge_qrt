"""Experiences incrementales de features pour le modele GPT V2.

Toutes les experiences utilisent les memes folds de dates que la V1. Les
agregats transversaux sont construits uniquement a partir de X et restent donc
disponibles de la meme facon dans le train et dans le test.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_validation import CVconfig, ModelConfig, run_cv  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


BASE_RETURNS = [
    "RET_1",
    "RET_2",
    "RET_3",
    "RET_4",
    "RET_7",
    "RET_8",
    "RET_9",
    "RET_18",
]
ALL_RETURNS = [f"RET_{i}" for i in range(1, 21)]
ALL_VOLUMES = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
CATEGORICAL_FEATURES = ["ALLOCATION", "GROUP"]


RETURN_DYNAMICS = [
    "ret_mean_3",
    "ret_mean_5",
    "ret_mean_10",
    "ret_mean_20",
    "ret_vol_5",
    "ret_vol_10",
    "ret_vol_20",
    "ret_downside_20",
    "ret_positive_share_5",
    "ret_positive_share_20",
    "ret_sign_changes_20",
    "ret_ac1_20",
    "ret_recent_vs_long",
    "ret_1_over_vol_20",
]

LIQUIDITY_SUMMARY = [
    "MEDIAN_DAILY_TURNOVER",
    "sv_mean_5",
    "sv_mean_20",
    "sv_abs_mean_5",
    "sv_abs_mean_20",
    "sv_vol_20",
    "sv_positive_share_20",
    "ret_sv_corr_20",
    "ret1_x_sv1",
    "ret1_x_turnover",
]

DATE_BASES = [
    "RET_1",
    "ret_mean_5",
    "ret_mean_20",
    "ret_vol_20",
    "SIGNED_VOLUME_1",
    "sv_mean_20",
    "MEDIAN_DAILY_TURNOVER",
]
DATE_REGIME = [
    f"date_{stat}_{column}"
    for column in DATE_BASES
    for stat in ("mean", "std", "z", "rank")
]
GROUP_REGIME = [
    f"group_date_{stat}_{column}"
    for column in DATE_BASES
    for stat in ("mean", "std", "z", "rank")
]


def _row_correlation(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    """Correlation ligne par ligne avec gestion des valeurs manquantes."""
    x = left.to_numpy(float)
    y = right.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=1)
    xx = np.where(valid, x, 0.0)
    yy = np.where(valid, y, 0.0)
    sx = xx.sum(axis=1)
    sy = yy.sum(axis=1)
    sxx = (xx * xx).sum(axis=1)
    syy = (yy * yy).sum(axis=1)
    sxy = (xx * yy).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = n * sxy - sx * sy
        denominator = np.sqrt((n * sxx - sx**2) * (n * syy - sy**2))
        result = numerator / denominator
    result[(n < 3) | ~np.isfinite(result)] = np.nan
    return result


def _add_cross_sectional_features(
    features: pd.DataFrame,
    groupers: list[pd.Series],
    prefix: str,
) -> pd.DataFrame:
    """Ajoute moyennes, dispersions, z-scores et rangs sans utiliser y."""
    grouped = features.groupby(groupers, observed=True)
    additions: dict[str, pd.Series] = {}
    for column in DATE_BASES:
        values = features[column]
        mean = grouped[column].transform("mean")
        std = grouped[column].transform("std")
        additions[f"{prefix}_mean_{column}"] = mean.astype("float32")
        additions[f"{prefix}_std_{column}"] = std.astype("float32")
        with np.errstate(invalid="ignore", divide="ignore"):
            zscore = (values - mean) / std.replace(0.0, np.nan)
        additions[f"{prefix}_z_{column}"] = zscore.astype("float32")
        additions[f"{prefix}_rank_{column}"] = grouped[column].rank(
            pct=True,
            method="average",
        ).astype("float32")
    return pd.concat([features, pd.DataFrame(additions, index=features.index)], axis=1)


def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """Construit toutes les features candidates legitimes en une seule passe."""
    features = X[["TS", "ALLOCATION", "GROUP", "MEDIAN_DAILY_TURNOVER"]].copy()
    features[ALL_RETURNS + ALL_VOLUMES] = X[ALL_RETURNS + ALL_VOLUMES].astype(
        "float32"
    )

    returns = features[ALL_RETURNS]
    volumes = features[ALL_VOLUMES]

    for window in (3, 5, 10, 20):
        window_returns = returns[[f"RET_{i}" for i in range(1, window + 1)]]
        features[f"ret_mean_{window}"] = window_returns.mean(axis=1).astype("float32")
    for window in (5, 10, 20):
        window_returns = returns[[f"RET_{i}" for i in range(1, window + 1)]]
        features[f"ret_vol_{window}"] = window_returns.std(axis=1).astype("float32")

    features["ret_downside_20"] = np.sqrt(
        returns.where(returns < 0.0).pow(2).mean(axis=1)
    ).astype("float32")
    features["ret_positive_share_5"] = (returns.iloc[:, :5] > 0.0).mean(axis=1).astype("float32")
    features["ret_positive_share_20"] = (returns > 0.0).mean(axis=1).astype("float32")
    signs = np.sign(returns.to_numpy(float))
    valid_sign_pairs = np.isfinite(signs[:, 1:]) & np.isfinite(signs[:, :-1])
    sign_changes = np.where(
        valid_sign_pairs,
        signs[:, 1:] != signs[:, :-1],
        np.nan,
    )
    features["ret_sign_changes_20"] = np.nanmean(sign_changes, axis=1).astype("float32")
    features["ret_ac1_20"] = _row_correlation(
        returns.iloc[:, :-1],
        returns.iloc[:, 1:],
    ).astype("float32")
    features["ret_recent_vs_long"] = (
        features["ret_mean_5"] - features["ret_mean_20"]
    ).astype("float32")
    features["ret_1_over_vol_20"] = (
        features["RET_1"] / features["ret_vol_20"].replace(0.0, np.nan)
    ).astype("float32")

    features["sv_mean_5"] = volumes.iloc[:, :5].mean(axis=1).astype("float32")
    features["sv_mean_20"] = volumes.mean(axis=1).astype("float32")
    features["sv_abs_mean_5"] = volumes.iloc[:, :5].abs().mean(axis=1).astype("float32")
    features["sv_abs_mean_20"] = volumes.abs().mean(axis=1).astype("float32")
    features["sv_vol_20"] = volumes.std(axis=1).astype("float32")
    features["sv_positive_share_20"] = (volumes > 0.0).mean(axis=1).astype("float32")
    features["ret_sv_corr_20"] = _row_correlation(returns, volumes).astype("float32")
    features["ret1_x_sv1"] = (features["RET_1"] * features["SIGNED_VOLUME_1"]).astype("float32")
    features["ret1_x_turnover"] = (
        features["RET_1"] * features["MEDIAN_DAILY_TURNOVER"]
    ).astype("float32")

    features = _add_cross_sectional_features(features, [features["TS"]], "date")
    features = _add_cross_sectional_features(
        features,
        [features["TS"], features["GROUP"]],
        "group_date",
    )
    date_groups = features.groupby("TS", observed=True)
    allocation_groups = features.groupby(["TS", "GROUP"], observed=True)
    availability_features: dict[str, pd.Series] = {
        "date_count_SIGNED_VOLUME_1": date_groups["SIGNED_VOLUME_1"]
        .transform("count")
        .astype("float32"),
        "group_date_count_SIGNED_VOLUME_1": allocation_groups["SIGNED_VOLUME_1"]
        .transform("count")
        .astype("float32"),
    }
    for lag in (2, 3, 4, 5):
        column = f"SIGNED_VOLUME_{lag}"
        availability_features[f"group_date_std_{column}"] = allocation_groups[
            column
        ].transform("std").astype("float32")
    features = pd.concat(
        [features, pd.DataFrame(availability_features, index=features.index)],
        axis=1,
    )
    return features


EXPERIMENTS: dict[str, list[str]] = {
    "baseline": BASE_RETURNS,
    "all_returns": ALL_RETURNS,
    "return_dynamics": ALL_RETURNS + RETURN_DYNAMICS,
    "liquidity_summary": BASE_RETURNS + LIQUIDITY_SUMMARY,
    "liquidity_full": BASE_RETURNS + ALL_VOLUMES + LIQUIDITY_SUMMARY,
    "date_regime": BASE_RETURNS + RETURN_DYNAMICS + LIQUIDITY_SUMMARY + DATE_REGIME,
    "group_regime": BASE_RETURNS + RETURN_DYNAMICS + LIQUIDITY_SUMMARY + GROUP_REGIME,
    "combined": (
        ALL_RETURNS
        + RETURN_DYNAMICS
        + ALL_VOLUMES
        + LIQUIDITY_SUMMARY
        + DATE_REGIME
        + GROUP_REGIME
    ),
}


@dataclass
class SparseRidgeDesign:
    """Matrice sparse generique avec effets fixes et pente locale RET_1."""

    numeric_features: list[str]

    def __post_init__(self) -> None:
        self.numeric_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
        self.category_encoder = OneHotEncoder(handle_unknown="ignore")
        self.allocation_encoder = OneHotEncoder(handle_unknown="ignore")

    def fit(self, X: pd.DataFrame) -> "SparseRidgeDesign":
        self.numeric_pipeline.fit(X[self.numeric_features])
        self.category_encoder.fit(X[CATEGORICAL_FEATURES])
        self.allocation_encoder.fit(X[["ALLOCATION"]])
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        numeric = self.numeric_pipeline.transform(X[self.numeric_features])
        categories = self.category_encoder.transform(X[CATEGORICAL_FEATURES])
        allocation = self.allocation_encoder.transform(X[["ALLOCATION"]])
        ret1_position = self.numeric_features.index("RET_1")
        allocation_ret1 = allocation.multiply(numeric[:, ret1_position, None])
        return sparse.hstack(
            [sparse.csr_matrix(numeric), categories, allocation_ret1],
            format="csr",
        )


def make_preprocessor(
    numeric_features: list[str],
) -> Callable[[pd.DataFrame, pd.DataFrame], tuple[sparse.csr_matrix, sparse.csr_matrix]]:
    """Cree le callback de preprocessing attendu par le framework du projet."""
    def preprocess(
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        design = SparseRidgeDesign(numeric_features).fit(X_train)
        return design.transform(X_train), design.transform(X_valid)

    return preprocess


def score_oof(oof: pd.DataFrame) -> dict[str, float]:
    """Retourne les mesures communes aux experiences."""
    accuracy = float(oof["is_correct"].mean())
    by_ts = oof.groupby("TS", observed=True)["is_correct"].mean()
    by_allocation = oof.groupby("ALLOCATION", observed=True)["is_correct"].mean()
    return {
        "accuracy": accuracy,
        "ts_standard_error": float(by_ts.std() / np.sqrt(len(by_ts))),
        "ts_penalized_score": float(accuracy - by_ts.std() / np.sqrt(len(by_ts))),
        "allocation_standard_error": float(
            by_allocation.std() / np.sqrt(len(by_allocation))
        ),
    }


def run_experiments(
    names: list[str],
    output_dir: Path,
    n_splits: int = 8,
    random_state: int = 0,
    alpha: float = 100.0,
) -> pd.DataFrame:
    """Execute les experiences demandees et sauvegarde leurs sorties OOF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    X = build_features(X_raw)
    result_rows: list[dict[str, float | str | int]] = []
    fold_rows: list[dict[str, float | str | int]] = []

    for name in names:
        numeric_features = EXPERIMENTS[name]
        model_config = ModelConfig(
            model=Ridge(alpha=alpha, solver="lsqr"),
            regression=True,
            features=CATEGORICAL_FEATURES + numeric_features,
            preprocessing=make_preprocessor(numeric_features),
        )
        output = run_cv(
            X,
            y,
            model_config,
            CVconfig(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
                verbose=False,
            ),
        )
        scores = score_oof(output.oof_results)
        row: dict[str, float | str | int] = {
            "experiment": name,
            "n_numeric_features": len(numeric_features),
            "alpha": alpha,
            **scores,
        }
        result_rows.append(row)
        for fold in output.fold_results:
            fold_rows.append(
                {
                    "experiment": name,
                    "fold": fold.fold_id,
                    "accuracy": float(fold.valid_metrics["accuracy"]),
                    "mcc": float(fold.valid_metrics["mcc"]),
                    "mse": float(fold.valid_metrics["mse"]),
                }
            )
        oof = output.oof_results[[
            "fold",
            "TS",
            "ALLOCATION",
            "y_true",
            "y_true_binarized",
            "score",
            "prediction",
            "is_correct",
        ]].copy()
        oof.index.name = "ROW_ID"
        oof.to_csv(output_dir / f"oof_{name}.csv")

        pd.DataFrame(result_rows).to_csv(
            output_dir / "experiment_results.csv",
            index=False,
        )
        pd.DataFrame(fold_rows).to_csv(
            output_dir / "fold_results.csv",
            index=False,
        )
        print(json.dumps(row, ensure_ascii=False))

    return pd.DataFrame(result_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "v2_experiments",
    )
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=100.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    table = run_experiments(
        names=args.experiments,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        alpha=args.alpha,
    )
    print(table.to_string(index=False))

"""Tests de formes non lineaires suggerees par l'audit de quantiles.

Les transformations sont des rangs, polynomes et fonctions charnieres de X.
Les moyennes conditionnelles de cible servent uniquement au diagnostic et ne
sont jamais injectees comme variables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import (  # noqa: E402
    BASE_RETURNS,
    CATEGORICAL_FEATURES,
    SparseRidgeDesign,
)
from gpt.v3_features import build_v3_features  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


TURNOVER_SHAPE = [
    "turnover_group_rank",
    "turnover_group_rank_hi80",
    "ret1_x_turnover_group_rank",
    "ret1_x_turnover_group_rank_hi80",
]
DATE_DISPERSION_SHAPE = [
    "date_sv1_std_percentile",
    "date_sv1_std_h20",
    "date_sv1_std_h40",
    "date_sv1_std_h60",
    "date_sv1_std_h80",
]
GROUP_DISPERSION_SHAPE = [
    "group_date_sv1_std_percentile",
    "group_date_sv1_std_h20",
    "group_date_sv1_std_h40",
    "group_date_sv1_std_h60",
    "group_date_sv1_std_h80",
]
SV_POSITIVE_SHAPE = [
    "sv_positive_share_20",
    "sv_positive_share_20_sq",
    "sv_positive_share_hi80",
    "sv_all_positive",
]


FEATURE_SETS = {
    "baseline": BASE_RETURNS,
    "turnover_rank_linear": BASE_RETURNS + ["turnover_group_rank"],
    "turnover_rank_shape": BASE_RETURNS + TURNOVER_SHAPE,
    "date_sv_dispersion_shape": BASE_RETURNS + DATE_DISPERSION_SHAPE,
    "group_sv_dispersion_shape": BASE_RETURNS + GROUP_DISPERSION_SHAPE,
    "sv_positive_shape": BASE_RETURNS + SV_POSITIVE_SHAPE,
    "quantile_shape_combo": (
        BASE_RETURNS
        + TURNOVER_SHAPE
        + DATE_DISPERSION_SHAPE
        + GROUP_DISPERSION_SHAPE
        + SV_POSITIVE_SHAPE
    ),
}


def repeated_group_percentile(
    frame: pd.DataFrame,
    keys: list[str],
    value: str,
    within: str | None = None,
) -> pd.Series:
    """Rang de chaque unite unique, remappe sur les lignes originales."""
    units = frame[keys + [value]].drop_duplicates(keys).copy()
    if within is None:
        units["percentile"] = units[value].rank(pct=True, method="average")
    else:
        units["percentile"] = units.groupby(within, observed=True)[value].rank(
            pct=True,
            method="average",
        )
    index = pd.MultiIndex.from_frame(units[keys])
    mapping = pd.Series(units["percentile"].to_numpy(), index=index)
    row_index = pd.MultiIndex.from_frame(frame[keys])
    return pd.Series(mapping.reindex(row_index).to_numpy(), index=frame.index)


def add_hinges(features: pd.DataFrame, source: str, prefix: str) -> None:
    """Ajoute une base lineaire par morceaux sur [0, 1]."""
    for knot in (0.2, 0.4, 0.6, 0.8):
        features[f"{prefix}_h{int(knot * 100)}"] = (
            features[source] - knot
        ).clip(lower=0.0).astype("float32")


def build_shape_features(X: pd.DataFrame) -> pd.DataFrame:
    """Construit les transformations candidates uniquement depuis X."""
    features, _ = build_v3_features(X)
    turnover_rank = features["group_date_rank_MEDIAN_DAILY_TURNOVER"]
    features["turnover_group_rank"] = turnover_rank.astype("float32")
    features["turnover_group_rank_hi80"] = (turnover_rank - 0.8).clip(
        lower=0.0
    ).astype("float32")
    features["ret1_x_turnover_group_rank"] = (
        features["RET_1"] * turnover_rank
    ).astype("float32")
    features["ret1_x_turnover_group_rank_hi80"] = (
        features["RET_1"] * features["turnover_group_rank_hi80"]
    ).astype("float32")

    features["date_sv1_std_percentile"] = repeated_group_percentile(
        features,
        ["TS"],
        "date_std_SIGNED_VOLUME_1",
    ).astype("float32")
    features["group_date_sv1_std_percentile"] = repeated_group_percentile(
        features,
        ["TS", "GROUP"],
        "group_date_std_SIGNED_VOLUME_1",
        within="GROUP",
    ).astype("float32")
    add_hinges(features, "date_sv1_std_percentile", "date_sv1_std")
    add_hinges(
        features,
        "group_date_sv1_std_percentile",
        "group_date_sv1_std",
    )

    share = features["sv_positive_share_20"]
    features["sv_positive_share_20_sq"] = share.pow(2).astype("float32")
    features["sv_positive_share_hi80"] = (share - 0.8).clip(lower=0.0).astype(
        "float32"
    )
    features["sv_all_positive"] = share.eq(1.0).astype("float32")
    return features


def run_tests(
    names: list[str],
    regularizations: list[float],
    max_folds: int,
    random_state: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Evalue les bases de fonctions sur des folds de dates fixes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = build_shape_features(ChallengeDataLoader.load_X_train())
    y = ChallengeDataLoader.load_y_train()
    unique_ts = X["TS"].unique()
    splits = KFold(n_splits=8, shuffle=True, random_state=random_state).split(unique_ts)
    states = {
        (name, c): {
            "prediction": pd.Series(np.nan, index=X.index),
            "fold": pd.Series(np.nan, index=X.index),
            "fold_rows": [],
        }
        for name in names
        for c in regularizations
    }

    for fold, (train_date_idx, valid_date_idx) in enumerate(splits, start=1):
        if fold > max_folds:
            break
        train_mask = X["TS"].isin(unique_ts[train_date_idx])
        valid_mask = X["TS"].isin(unique_ts[valid_date_idx])
        for name in names:
            numeric = FEATURE_SETS[name]
            columns = CATEGORICAL_FEATURES + numeric
            design = SparseRidgeDesign(numeric).fit(X.loc[train_mask, columns])
            train_matrix = design.transform(X.loc[train_mask, columns])
            valid_matrix = design.transform(X.loc[valid_mask, columns])
            for c in regularizations:
                model = LogisticRegression(
                    C=c,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=200,
                    tol=1e-5,
                    random_state=0,
                )
                model.fit(train_matrix, y.loc[train_mask, "target_binarized"])
                prediction = (model.predict_proba(valid_matrix)[:, 1] > 0.5).astype(
                    "int8"
                )
                accuracy = accuracy_score(
                    y.loc[valid_mask, "target_binarized"], prediction
                )
                state = states[(name, c)]
                state["prediction"].loc[valid_mask] = prediction
                state["fold"].loc[valid_mask] = fold
                state["fold_rows"].append(
                    {
                        "experiment": name,
                        "C": c,
                        "fold": fold,
                        "accuracy": accuracy,
                    }
                )
                print(
                    f"{name} C={c:g} fold={fold} accuracy={accuracy:.6f}",
                    flush=True,
                )

    results = []
    fold_rows = []
    for (name, c), state in states.items():
        covered = state["prediction"].notna()
        correct = state["prediction"].loc[covered].astype("int8").eq(
            y.loc[covered, "target_binarized"].astype("int8")
        )
        by_ts = correct.groupby(X.loc[covered, "TS"]).mean()
        by_allocation = correct.groupby(X.loc[covered, "ALLOCATION"]).mean()
        accuracy = float(correct.mean())
        results.append(
            {
                "experiment": name,
                "C": c,
                "n_folds": len(state["fold_rows"]),
                "n_oof_rows": int(covered.sum()),
                "accuracy": accuracy,
                "ts_standard_error": float(by_ts.std() / np.sqrt(len(by_ts))),
                "ts_penalized_score": float(
                    accuracy - by_ts.std() / np.sqrt(len(by_ts))
                ),
                "allocation_standard_error": float(
                    by_allocation.std() / np.sqrt(len(by_allocation))
                ),
            }
        )
        fold_rows.extend(state["fold_rows"])
        pd.DataFrame(
            {
                "fold": state["fold"].loc[covered],
                "TS": X.loc[covered, "TS"],
                "ALLOCATION": X.loc[covered, "ALLOCATION"],
                "y_true_binarized": y.loc[covered, "target_binarized"],
                "prediction": state["prediction"].loc[covered],
                "is_correct": correct.astype("int8"),
            }
        ).rename_axis("ROW_ID").to_csv(output_dir / f"oof_{name}_C_{c:g}.csv")

    table = pd.DataFrame(results).sort_values(
        ["accuracy", "ts_penalized_score"], ascending=False
    )
    table.to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_results.csv", index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(FEATURE_SETS),
        default=list(FEATURE_SETS),
    )
    parser.add_argument("--regularizations", nargs="+", type=float, default=[0.003])
    parser.add_argument("--max-folds", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "quantile_shape_screen",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_tests(
        args.experiments,
        args.regularizations,
        args.max_folds,
        args.random_state,
        args.output_dir,
    )
    print(result.to_string(index=False))

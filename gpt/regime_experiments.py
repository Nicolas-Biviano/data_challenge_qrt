"""Modeles hierarchiques de regime par date et par groupe.

Chaque regime est predit uniquement a partir d'agregats de X. Les dates de
validation sont absentes de l'entrainement et aucun target encoding n'est
utilise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import ChallengeDataLoader  # noqa: E402


RETURNS = [f"RET_{i}" for i in range(1, 21)]
VOLUMES = [f"SIGNED_VOLUME_{i}" for i in range(1, 6)]
AGGREGATED_COLUMNS = RETURNS + VOLUMES + ["MEDIAN_DAILY_TURNOVER"]


def aggregate_X(X: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Agrege les empreintes historiques sans consulter la cible."""
    grouped = X.groupby(group_columns, observed=True)
    moments = grouped[AGGREGATED_COLUMNS].agg(["mean", "std"])
    moments.columns = [f"{stat}_{column}" for column, stat in moments.columns]
    sign_share = grouped[RETURNS].agg(lambda values: (values > 0.0).mean())
    sign_share.columns = [f"positive_share_{column}" for column in sign_share.columns]
    size = grouped.size().rename("n_allocations")
    return pd.concat([moments, sign_share, size], axis=1).astype("float32")


def build_regime_tables(
    X: pd.DataFrame,
    y: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construit les tables date et date-groupe ainsi que leurs cibles."""
    date_X = aggregate_X(X, ["TS"])
    group_X = aggregate_X(X, ["TS", "GROUP"])
    global_for_group = date_X.add_prefix("global_").reindex(
        group_X.index.get_level_values("TS")
    )
    global_for_group.index = group_X.index
    group_dummies = pd.get_dummies(
        group_X.index.get_level_values("GROUP"),
        prefix="GROUP",
        dtype="float32",
    )
    group_dummies.index = group_X.index
    group_X = pd.concat([group_X, global_for_group, group_dummies], axis=1)

    joined = X[["TS", "GROUP"]].join(y[["target", "target_binarized"]])
    date_y = joined.groupby("TS", observed=True).agg(
        target=("target", "mean"),
        positive_rate=("target_binarized", "mean"),
        n_rows=("target", "size"),
    )
    group_y = joined.groupby(["TS", "GROUP"], observed=True).agg(
        target=("target", "mean"),
        positive_rate=("target_binarized", "mean"),
        n_rows=("target", "size"),
    )
    return date_X, date_y, group_X, group_y


def make_model(name: str):
    """Modeles regularises adaptes au faible nombre de dates."""
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=100.0),
        )
    if name == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=40,
            l2_regularization=20.0,
            max_bins=64,
            early_stopping=False,
            random_state=0,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            max_features=0.7,
            n_jobs=-1,
            random_state=0,
        )
    raise ValueError(f"Modele inconnu: {name}")


def fit_with_weights(model, X, y, weights):
    """Transmet les poids au dernier estimateur d'une pipeline si necessaire."""
    if hasattr(model, "named_steps"):
        model.fit(X, y, ridge__sample_weight=weights)
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def run_one(
    level: str,
    model_name: str,
    target_name: str,
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    table_X: pd.DataFrame,
    table_y: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str | float | int]:
    """Produit des predictions OOF de regime puis les diffuse aux lignes."""
    unique_ts = X_raw["TS"].unique()
    splitter = KFold(n_splits=8, shuffle=True, random_state=0)
    regime_score = pd.Series(np.nan, index=table_X.index, dtype="float64")
    regime_fold = pd.Series(np.nan, index=table_X.index, dtype="float64")

    for fold, (train_date_idx, valid_date_idx) in enumerate(
        splitter.split(unique_ts),
        start=1,
    ):
        train_dates = unique_ts[train_date_idx]
        valid_dates = unique_ts[valid_date_idx]
        if level == "date":
            train_mask = table_X.index.isin(train_dates)
            valid_mask = table_X.index.isin(valid_dates)
        else:
            dates = table_X.index.get_level_values("TS")
            train_mask = dates.isin(train_dates)
            valid_mask = dates.isin(valid_dates)

        model = make_model(model_name)
        fit_with_weights(
            model,
            table_X.loc[train_mask],
            table_y.loc[train_mask, target_name],
            table_y.loc[train_mask, "n_rows"],
        )
        regime_score.loc[valid_mask] = model.predict(table_X.loc[valid_mask])
        regime_fold.loc[valid_mask] = fold

    mapping = pd.DataFrame(
        {"regime_score": regime_score, "fold": regime_fold},
        index=table_X.index,
    )
    row_keys = X_raw[["TS", "GROUP"]].copy()
    if level == "date":
        row_score = row_keys["TS"].map(mapping["regime_score"])
        row_fold = row_keys["TS"].map(mapping["fold"])
    else:
        row_index = pd.MultiIndex.from_frame(row_keys[["TS", "GROUP"]])
        row_score = pd.Series(
            mapping["regime_score"].reindex(row_index).to_numpy(),
            index=X_raw.index,
        )
        row_fold = pd.Series(
            mapping["fold"].reindex(row_index).to_numpy(),
            index=X_raw.index,
        )

    threshold = 0.0 if target_name == "target" else 0.5
    row_prediction = row_score > threshold
    correct = row_prediction == y_raw["target_binarized"].astype(bool)
    by_ts = correct.groupby(X_raw["TS"]).mean()
    accuracy = float(correct.mean())
    experiment = f"{level}_{model_name}_{target_name}"
    oof = pd.DataFrame(
        {
            "fold": row_fold,
            "TS": X_raw["TS"],
            "ALLOCATION": X_raw["ALLOCATION"],
            "GROUP": X_raw["GROUP"],
            "y_true": y_raw["target"],
            "y_true_binarized": y_raw["target_binarized"],
            "score": row_score,
            "prediction": row_prediction.astype("int8"),
            "is_correct": correct.astype("int8"),
        },
        index=X_raw.index,
    )
    oof.index.name = "ROW_ID"
    oof.to_csv(output_dir / f"oof_{experiment}.csv")
    result: dict[str, str | float | int] = {
        "experiment": experiment,
        "level": level,
        "model": model_name,
        "target": target_name,
        "n_training_units": len(table_X),
        "n_features": table_X.shape[1],
        "accuracy": accuracy,
        "ts_standard_error": float(by_ts.std() / np.sqrt(len(by_ts))),
        "ts_penalized_score": float(accuracy - by_ts.std() / np.sqrt(len(by_ts))),
    }
    print(result, flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", choices=["date", "group_date"], default=["date", "group_date"])
    parser.add_argument("--models", nargs="+", choices=["ridge", "hgb", "extra_trees"], default=["ridge", "hgb", "extra_trees"])
    parser.add_argument("--targets", nargs="+", choices=["target", "positive_rate"], default=["target", "positive_rate"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "regime_experiments",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X_raw = ChallengeDataLoader.load_X_train()
    y_raw = ChallengeDataLoader.load_y_train()
    date_X, date_y, group_X, group_y = build_regime_tables(X_raw, y_raw)
    results = []
    for level in args.levels:
        table_X, table_y = (date_X, date_y) if level == "date" else (group_X, group_y)
        for model_name in args.models:
            for target_name in args.targets:
                results.append(
                    run_one(
                        level,
                        model_name,
                        target_name,
                        X_raw,
                        y_raw,
                        table_X,
                        table_y,
                        args.output_dir,
                    )
                )
                pd.DataFrame(results).to_csv(args.output_dir / "results.csv", index=False)
    print(pd.DataFrame(results).sort_values("accuracy", ascending=False).to_string(index=False))


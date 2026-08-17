"""Modeles non lineaires legitimes pour la V2, sans target encoding.

La branche arbre utilise uniquement des variables numeriques construites depuis
X et un one-hot du groupe. L'identifiant d'allocation n'est pas transforme a
partir de la cible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import (  # noqa: E402
    ALL_VOLUMES,
    BASE_RETURNS,
    DATE_REGIME,
    GROUP_REGIME,
    LIQUIDITY_SUMMARY,
    RETURN_DYNAMICS,
    build_features,
)
from src.dataloader import ChallengeDataLoader  # noqa: E402


TREE_FEATURES = list(
    dict.fromkeys(
        BASE_RETURNS
        + RETURN_DYNAMICS
        + ALL_VOLUMES[:5]
        + LIQUIDITY_SUMMARY
        + DATE_REGIME
        + GROUP_REGIME
    )
)


def prepare_tree_matrix(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Construit une matrice dense float32 independante de la cible."""
    features = build_features(X_raw)
    matrix = features[TREE_FEATURES].astype("float32")
    group_dummies = pd.get_dummies(
        features["GROUP"],
        prefix="GROUP",
        dtype="float32",
    )
    return pd.concat([matrix, group_dummies], axis=1)


def make_model(name: str, max_iter: int):
    """Retourne un boosting volontairement regularise."""
    common = {
        "learning_rate": 0.05,
        "max_iter": max_iter,
        "max_leaf_nodes": 15,
        "max_depth": None,
        "min_samples_leaf": 500,
        "l2_regularization": 10.0,
        "max_bins": 64,
        "early_stopping": False,
        "random_state": 0,
    }
    if name == "hgb_regression":
        return HistGradientBoostingRegressor(loss="squared_error", **common)
    if name == "hgb_classification":
        return HistGradientBoostingClassifier(loss="log_loss", **common)
    raise ValueError(f"Modele inconnu: {name}")


def run_model(
    name: str,
    X_raw: pd.DataFrame,
    X_matrix: pd.DataFrame,
    y: pd.DataFrame,
    output_dir: Path,
    max_iter: int,
    max_folds: int,
) -> dict[str, float | int | str]:
    """Execute des folds de dates fixes et exporte les predictions."""
    unique_ts = X_raw["TS"].unique()
    splitter = KFold(n_splits=8, shuffle=True, random_state=0)
    prediction = pd.Series(np.nan, index=X_raw.index, dtype="float64")
    score = pd.Series(np.nan, index=X_raw.index, dtype="float64")
    fold_id = pd.Series(np.nan, index=X_raw.index, dtype="float64")
    fold_rows = []

    for current_fold, (train_date_idx, valid_date_idx) in enumerate(
        splitter.split(unique_ts),
        start=1,
    ):
        if current_fold > max_folds:
            break
        train_dates = unique_ts[train_date_idx]
        valid_dates = unique_ts[valid_date_idx]
        train_mask = X_raw["TS"].isin(train_dates)
        valid_mask = X_raw["TS"].isin(valid_dates)
        model = make_model(name, max_iter=max_iter)

        if name == "hgb_regression":
            model.fit(X_matrix.loc[train_mask], y.loc[train_mask, "target"])
            local_score = model.predict(X_matrix.loc[valid_mask])
            local_prediction = local_score > 0.0
        else:
            model.fit(
                X_matrix.loc[train_mask],
                y.loc[train_mask, "target_binarized"],
            )
            local_score = model.predict_proba(X_matrix.loc[valid_mask])[:, 1]
            local_prediction = local_score > 0.5

        local_accuracy = accuracy_score(
            y.loc[valid_mask, "target_binarized"],
            local_prediction,
        )
        prediction.loc[valid_mask] = local_prediction.astype("int8")
        score.loc[valid_mask] = local_score
        fold_id.loc[valid_mask] = current_fold
        fold_rows.append(
            {
                "model": name,
                "fold": current_fold,
                "accuracy": local_accuracy,
                "n_train": int(train_mask.sum()),
                "n_valid": int(valid_mask.sum()),
            }
        )
        print(f"{name} fold={current_fold} accuracy={local_accuracy:.6f}", flush=True)

    covered = prediction.notna()
    correct = (
        prediction.loc[covered].astype("int8")
        == y.loc[covered, "target_binarized"].astype("int8")
    )
    by_ts = correct.groupby(X_raw.loc[covered, "TS"]).mean()
    accuracy = float(correct.mean())
    result: dict[str, float | int | str] = {
        "model": name,
        "max_iter": max_iter,
        "n_features": int(X_matrix.shape[1]),
        "n_folds": len(fold_rows),
        "n_oof_rows": int(covered.sum()),
        "accuracy": accuracy,
        "ts_standard_error": float(by_ts.std() / np.sqrt(len(by_ts))),
        "ts_penalized_score": float(accuracy - by_ts.std() / np.sqrt(len(by_ts))),
    }
    oof = pd.DataFrame(
        {
            "fold": fold_id,
            "TS": X_raw["TS"],
            "ALLOCATION": X_raw["ALLOCATION"],
            "y_true": y["target"],
            "y_true_binarized": y["target_binarized"],
            "score": score,
            "prediction": prediction,
        },
        index=X_raw.index,
    )
    oof["is_correct"] = np.where(
        covered,
        (oof["prediction"] == oof["y_true_binarized"]).astype("float64"),
        np.nan,
    )
    oof.index.name = "ROW_ID"
    oof.to_csv(output_dir / f"oof_{name}.csv")
    pd.DataFrame(fold_rows).to_csv(output_dir / f"folds_{name}.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["hgb_regression", "hgb_classification"],
        default=["hgb_regression", "hgb_classification"],
    )
    parser.add_argument("--max-iter", type=int, default=75)
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "nonlinear_screen",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X_raw = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    X_matrix = prepare_tree_matrix(X_raw)
    results = [
        run_model(
            name=name,
            X_raw=X_raw,
            X_matrix=X_matrix,
            y=y,
            output_dir=args.output_dir,
            max_iter=args.max_iter,
            max_folds=args.max_folds,
        )
        for name in args.models
    ]
    pd.DataFrame(results).to_csv(args.output_dir / "results.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))


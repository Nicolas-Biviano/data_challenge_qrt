"""Classification directe sur le design sparse de la V1.

Ce script compare plusieurs regularisations logistiques sur les memes folds de
TS. Les categories sont encodees en one-hot, sans target encoding.
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
    build_features,
)
from src.dataloader import ChallengeDataLoader  # noqa: E402


FEATURE_SETS = {
    "baseline": BASE_RETURNS,
    "group_volume": BASE_RETURNS + ["group_date_std_SIGNED_VOLUME_1"],
}


def run_grid(
    regularizations: list[float],
    max_folds: int,
    output_dir: Path,
    feature_set: str,
) -> pd.DataFrame:
    """Evalue les valeurs de C en reutilisant le design de chaque fold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw = ChallengeDataLoader.load_X_train()
    X = build_features(X_raw) if feature_set != "baseline" else X_raw
    y = ChallengeDataLoader.load_y_train()
    numeric_features = FEATURE_SETS[feature_set]
    unique_ts = X["TS"].unique()
    splitter = KFold(n_splits=8, shuffle=True, random_state=0)
    states = {
        c: {
            "score": pd.Series(np.nan, index=X.index),
            "prediction": pd.Series(np.nan, index=X.index),
            "fold": pd.Series(np.nan, index=X.index),
            "fold_rows": [],
        }
        for c in regularizations
    }

    for fold, (train_date_idx, valid_date_idx) in enumerate(
        splitter.split(unique_ts),
        start=1,
    ):
        if fold > max_folds:
            break
        train_mask = X["TS"].isin(unique_ts[train_date_idx])
        valid_mask = X["TS"].isin(unique_ts[valid_date_idx])
        selected_features = CATEGORICAL_FEATURES + numeric_features
        design = SparseRidgeDesign(numeric_features).fit(
            X.loc[train_mask, selected_features]
        )
        X_train = design.transform(X.loc[train_mask, selected_features])
        X_valid = design.transform(X.loc[valid_mask, selected_features])
        y_train = y.loc[train_mask, "target_binarized"]
        y_valid = y.loc[valid_mask, "target_binarized"]

        for c in regularizations:
            model = LogisticRegression(
                C=c,
                penalty="l2",
                solver="lbfgs",
                max_iter=150,
                tol=1e-5,
                random_state=0,
            )
            model.fit(X_train, y_train)
            local_score = model.predict_proba(X_valid)[:, 1]
            local_prediction = local_score > 0.5
            local_accuracy = accuracy_score(y_valid, local_prediction)
            states[c]["score"].loc[valid_mask] = local_score
            states[c]["prediction"].loc[valid_mask] = local_prediction.astype("int8")
            states[c]["fold"].loc[valid_mask] = fold
            states[c]["fold_rows"].append(
                {"C": c, "fold": fold, "accuracy": local_accuracy}
            )
            print(f"logistic C={c:g} fold={fold} accuracy={local_accuracy:.6f}", flush=True)

    result_rows = []
    all_fold_rows = []
    for c, state in states.items():
        covered = state["prediction"].notna()
        correct = (
            state["prediction"].loc[covered].astype("int8")
            == y.loc[covered, "target_binarized"].astype("int8")
        )
        by_ts = correct.groupby(X.loc[covered, "TS"]).mean()
        accuracy = float(correct.mean())
        row = {
            "feature_set": feature_set,
            "C": c,
            "n_folds": len(state["fold_rows"]),
            "n_oof_rows": int(covered.sum()),
            "accuracy": accuracy,
            "ts_standard_error": float(by_ts.std() / np.sqrt(len(by_ts))),
            "ts_penalized_score": float(accuracy - by_ts.std() / np.sqrt(len(by_ts))),
        }
        result_rows.append(row)
        all_fold_rows.extend(state["fold_rows"])
        oof = pd.DataFrame(
            {
                "fold": state["fold"],
                "TS": X["TS"],
                "ALLOCATION": X["ALLOCATION"],
                "y_true": y["target"],
                "y_true_binarized": y["target_binarized"],
                "score": state["score"],
                "prediction": state["prediction"],
            },
            index=X.index,
        )
        oof["is_correct"] = np.where(
            covered,
            (oof["prediction"] == oof["y_true_binarized"]).astype(float),
            np.nan,
        )
        oof.index.name = "ROW_ID"
        oof.to_csv(output_dir / f"oof_logistic_{feature_set}_C_{c:g}.csv")

    pd.DataFrame(result_rows).to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(output_dir / "fold_results.csv", index=False)
    return pd.DataFrame(result_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regularizations", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="baseline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "classification_screen",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_grid(
        args.regularizations,
        args.max_folds,
        args.output_dir,
        args.feature_set,
    )
    print(results.to_string(index=False))

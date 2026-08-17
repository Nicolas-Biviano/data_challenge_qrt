"""Quantile Regression Forest V3: moyenne, mediane et fallback de confiance."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.v3_features import allocation_clusters, build_v3_features  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


QUANTILES = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]


def sparse_design(
    features: pd.DataFrame,
    numeric_columns: list[str],
    train_mask: pd.Series,
    valid_mask: pd.Series,
    cluster_code: pd.Series,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Imputation fold-safe et one-hot X-only des categories."""
    imputer = SimpleImputer(strategy="median")
    train_numeric = imputer.fit_transform(features.loc[train_mask, numeric_columns])
    valid_numeric = imputer.transform(features.loc[valid_mask, numeric_columns])
    categories = pd.DataFrame(
        {
            "ALLOCATION": features["ALLOCATION"],
            "GROUP": features["GROUP"],
            "ALLOCATION_CLUSTER_X_ONLY": cluster_code.astype(str),
        },
        index=features.index,
    )
    encoder = OneHotEncoder(handle_unknown="ignore")
    train_categories = encoder.fit_transform(categories.loc[train_mask])
    valid_categories = encoder.transform(categories.loc[valid_mask])
    train_matrix = sparse.hstack(
        [sparse.csr_matrix(train_numeric), train_categories],
        format="csr",
    )
    valid_matrix = sparse.hstack(
        [sparse.csr_matrix(valid_numeric), valid_categories],
        format="csr",
    )
    return train_matrix, valid_matrix


def run_qrf(
    block_name: str,
    max_folds: int,
    n_estimators: int,
    output_dir: Path,
    logistic_oof_path: Path,
) -> pd.DataFrame:
    """Entraine une QRF par fold et evalue les strategies predefinies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    features, blocks = build_v3_features(X)
    numeric_columns = blocks[block_name]
    logistic_oof = pd.read_csv(logistic_oof_path, index_col="ROW_ID").reindex(X.index)
    logistic_prediction = logistic_oof["prediction"].astype(bool)
    unique_ts = X["TS"].unique()
    splits = list(KFold(n_splits=8, shuffle=True, random_state=0).split(unique_ts))
    prediction_store: dict[str, pd.Series] = {
        name: pd.Series(np.nan, index=X.index, dtype="float64")
        for name in ["mean", "median", "fallback_40_60", "fallback_25_75", "fallback_10_90"]
    }
    score_store = pd.DataFrame(np.nan, index=X.index, columns=[f"q{q:g}" for q in QUANTILES])
    score_store["mean"] = np.nan
    fold_id = pd.Series(np.nan, index=X.index)
    fold_rows = []
    started = time.perf_counter()

    for fold, (train_date_idx, valid_date_idx) in enumerate(splits, start=1):
        if fold > max_folds:
            break
        train_mask = X["TS"].isin(unique_ts[train_date_idx])
        valid_mask = X["TS"].isin(unique_ts[valid_date_idx])
        cluster_code = allocation_clusters(
            X.loc[train_mask],
            X["ALLOCATION"],
            n_clusters=8,
        )
        train_matrix, valid_matrix = sparse_design(
            features,
            numeric_columns,
            train_mask,
            valid_mask,
            cluster_code,
        )
        model = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_leaf=500,
            max_features=0.5,
            bootstrap=True,
            max_samples=0.7,
            max_samples_leaf=256,
            n_jobs=-1,
            random_state=fold,
        )
        model.fit(train_matrix, y.loc[train_mask, "target"])
        quantile_scores = model.predict(valid_matrix, quantiles=QUANTILES)
        mean_score = model.predict(valid_matrix, quantiles="mean")
        local_index = X.index[valid_mask]
        local_quantiles = pd.DataFrame(
            quantile_scores,
            index=local_index,
            columns=[f"q{q:g}" for q in QUANTILES],
        )
        score_store.loc[local_index, local_quantiles.columns] = local_quantiles
        score_store.loc[local_index, "mean"] = mean_score
        fold_id.loc[local_index] = fold

        strategies: dict[str, np.ndarray] = {
            "mean": mean_score > 0.0,
            "median": local_quantiles["q0.5"].to_numpy() > 0.0,
        }
        for lower, upper, name in [
            ("q0.4", "q0.6", "fallback_40_60"),
            ("q0.25", "q0.75", "fallback_25_75"),
            ("q0.1", "q0.9", "fallback_10_90"),
        ]:
            confident_positive = local_quantiles[lower].to_numpy() > 0.0
            confident_negative = local_quantiles[upper].to_numpy() < 0.0
            confident = confident_positive | confident_negative
            local_prediction = logistic_prediction.loc[local_index].to_numpy().copy()
            local_prediction[confident] = confident_positive[confident]
            strategies[name] = local_prediction
            fold_rows.append(
                {
                    "strategy": name,
                    "fold": fold,
                    "confidence_coverage": float(confident.mean()),
                    "confidence_accuracy": float(
                        np.mean(
                            local_prediction[confident]
                            == y.loc[local_index[confident], "target_binarized"].to_numpy()
                        )
                    )
                    if confident.any()
                    else np.nan,
                }
            )

        truth = y.loc[local_index, "target_binarized"].to_numpy()
        for name, local_prediction in strategies.items():
            prediction_store[name].loc[local_index] = local_prediction.astype("int8")
            local_accuracy = float(np.mean(local_prediction == truth))
            fold_rows.append(
                {
                    "strategy": name,
                    "fold": fold,
                    "accuracy": local_accuracy,
                }
            )
            print(f"qrf_{block_name} {name} fold={fold} accuracy={local_accuracy:.6f}", flush=True)

    results = []
    covered = fold_id.notna()
    for name, predictions in prediction_store.items():
        correct = (
            predictions.loc[covered].astype("int8")
            == y.loc[covered, "target_binarized"].astype("int8")
        )
        by_ts = correct.groupby(X.loc[covered, "TS"]).mean()
        accuracy = float(correct.mean())
        results.append(
            {
                "experiment": f"qrf_{block_name}_{name}",
                "block": block_name,
                "strategy": name,
                "n_numeric_features": len(numeric_columns),
                "n_folds": max_folds,
                "n_estimators": n_estimators,
                "accuracy": accuracy,
                "ts_standard_error": float(by_ts.std() / np.sqrt(len(by_ts))),
                "ts_penalized_score": float(
                    accuracy - by_ts.std() / np.sqrt(len(by_ts))
                ),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
    export = pd.concat(
        [
            pd.DataFrame({"fold": fold_id, "TS": X["TS"], "ALLOCATION": X["ALLOCATION"]}),
            score_store.add_prefix("score_"),
            pd.DataFrame(
                {f"prediction_{name}": values for name, values in prediction_store.items()}
            ),
        ],
        axis=1,
    )
    export.index.name = "ROW_ID"
    export.to_csv(output_dir / f"oof_qrf_{block_name}.csv")
    pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_results.csv", index=False)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=["returns", "combined"], default="returns")
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "v3_qrf_screen",
    )
    parser.add_argument(
        "--logistic-oof",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "classification_full"
        / "oof_logistic_C_0.003.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    table = run_qrf(
        block_name=args.block,
        max_folds=args.max_folds,
        n_estimators=args.n_estimators,
        output_dir=args.output_dir,
        logistic_oof_path=args.logistic_oof,
    )
    print(table.sort_values("accuracy", ascending=False).to_string(index=False))


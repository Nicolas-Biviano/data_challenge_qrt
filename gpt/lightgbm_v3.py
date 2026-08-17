"""Screening LightGBM classique contre feuilles lineaires pour la V3."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.v3_features import allocation_clusters, build_v3_features  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


def standardized_matrices(
    features: pd.DataFrame,
    numeric_columns: list[str],
    train_mask: pd.Series,
    valid_mask: pd.Series,
    allocation_code: np.ndarray,
    group_code: np.ndarray,
    cluster_code: np.ndarray,
    category_set: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Standardise les variables numeriques dans le fold en preservant les NaN."""
    train_numeric = features.loc[train_mask, numeric_columns].to_numpy(
        dtype="float32",
        copy=True,
    )
    valid_numeric = features.loc[valid_mask, numeric_columns].to_numpy(
        dtype="float32",
        copy=True,
    )
    means = np.nanmean(train_numeric, axis=0)
    scales = np.nanstd(train_numeric, axis=0)
    scales[(scales == 0.0) | ~np.isfinite(scales)] = 1.0
    train_numeric = (train_numeric - means) / scales
    valid_numeric = (valid_numeric - means) / scales

    if category_set == "full":
        category_arrays = [allocation_code, group_code, cluster_code]
        category_names = ["ALLOCATION", "GROUP", "ALLOCATION_CLUSTER_X_ONLY"]
    elif category_set == "cluster":
        category_arrays = [group_code, cluster_code]
        category_names = ["GROUP", "ALLOCATION_CLUSTER_X_ONLY"]
    elif category_set == "group":
        category_arrays = [group_code]
        category_names = ["GROUP"]
    else:
        raise ValueError(f"Categorie inconnue: {category_set}")
    train_categories = np.column_stack(
        [values[train_mask] for values in category_arrays]
    ).astype("float32")
    valid_categories = np.column_stack(
        [values[valid_mask] for values in category_arrays]
    ).astype("float32")
    train_matrix = np.column_stack([train_numeric, train_categories])
    valid_matrix = np.column_stack([valid_numeric, valid_categories])
    feature_names = numeric_columns + category_names
    categorical_indices = list(range(len(numeric_columns), len(feature_names)))
    return train_matrix, valid_matrix, feature_names, categorical_indices


def make_model(leaf_type: str, n_estimators: int) -> lgb.LGBMRegressor:
    """Parametrage commun pour une comparaison apples-to-apples."""
    parameters: dict[str, object] = {
        "objective": "regression_l2",
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "num_leaves": 16,
        "min_child_samples": 1000,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "max_bin": 63,
        "force_col_wise": True,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 0,
    }
    if leaf_type == "linear":
        parameters["linear_tree"] = True
        parameters["linear_lambda"] = 10.0
    return lgb.LGBMRegressor(**parameters)


def run_screen(
    blocks_to_run: list[str],
    leaf_types: list[str],
    max_folds: int,
    n_estimators: int,
    output_dir: Path,
    category_set: str,
) -> pd.DataFrame:
    """Execute le screening et sauvegarde chaque resultat des qu'il termine."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    features, blocks = build_v3_features(X)
    allocation_code = pd.Categorical(X["ALLOCATION"]).codes.astype("int32")
    group_code = pd.Categorical(X["GROUP"]).codes.astype("int32")
    unique_ts = X["TS"].unique()
    splits = list(KFold(n_splits=8, shuffle=True, random_state=0).split(unique_ts))
    results: list[dict[str, object]] = []
    fold_results: list[dict[str, object]] = []

    for block_name in blocks_to_run:
        numeric_columns = blocks[block_name]
        for leaf_type in leaf_types:
            experiment = f"lgbm_{leaf_type}_{block_name}_{category_set}"
            oof_score = pd.Series(np.nan, index=X.index, dtype="float64")
            oof_fold = pd.Series(np.nan, index=X.index, dtype="float64")
            importance_rows = []
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
                ).to_numpy()
                train_matrix, valid_matrix, feature_names, categorical_indices = (
                    standardized_matrices(
                        features,
                        numeric_columns,
                        train_mask,
                        valid_mask,
                        allocation_code,
                        group_code,
                        cluster_code,
                        category_set,
                    )
                )
                model = make_model(leaf_type, n_estimators=n_estimators)
                model.fit(
                    train_matrix,
                    y.loc[train_mask, "target"],
                    categorical_feature=categorical_indices,
                    feature_name=feature_names,
                )
                local_score = model.predict(valid_matrix)
                local_prediction = local_score > 0.0
                local_truth = y.loc[valid_mask, "target_binarized"].to_numpy()
                local_accuracy = float(np.mean(local_prediction == local_truth))
                oof_score.loc[valid_mask] = local_score
                oof_fold.loc[valid_mask] = fold
                fold_results.append(
                    {
                        "experiment": experiment,
                        "fold": fold,
                        "accuracy": local_accuracy,
                        "n_train": int(train_mask.sum()),
                        "n_valid": int(valid_mask.sum()),
                    }
                )
                importance_rows.append(
                    pd.DataFrame(
                        {
                            "experiment": experiment,
                            "fold": fold,
                            "feature": feature_names,
                            "split_importance": model.feature_importances_,
                        }
                    )
                )
                print(
                    f"{experiment} fold={fold} accuracy={local_accuracy:.6f}",
                    flush=True,
                )

            covered = oof_score.notna()
            correct = (
                (oof_score.loc[covered] > 0.0)
                == y.loc[covered, "target_binarized"].astype(bool)
            )
            by_ts = correct.groupby(X.loc[covered, "TS"]).mean()
            accuracy = float(correct.mean())
            row: dict[str, object] = {
                "experiment": experiment,
                "block": block_name,
                "leaf_type": leaf_type,
                "category_set": category_set,
                "n_numeric_features": len(numeric_columns),
                "n_folds": int(max_folds),
                "n_estimators": n_estimators,
                "accuracy": accuracy,
                "ts_standard_error": float(by_ts.std() / np.sqrt(len(by_ts))),
                "ts_penalized_score": float(
                    accuracy - by_ts.std() / np.sqrt(len(by_ts))
                ),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            results.append(row)
            oof = pd.DataFrame(
                {
                    "fold": oof_fold,
                    "TS": X["TS"],
                    "ALLOCATION": X["ALLOCATION"],
                    "y_true": y["target"],
                    "y_true_binarized": y["target_binarized"],
                    "score": oof_score,
                },
                index=X.index,
            )
            oof["prediction"] = np.where(covered, (oof_score > 0.0).astype(float), np.nan)
            oof["is_correct"] = np.where(
                covered,
                (oof["prediction"] == oof["y_true_binarized"]).astype(float),
                np.nan,
            )
            oof.index.name = "ROW_ID"
            oof.to_csv(output_dir / f"oof_{experiment}.csv")
            pd.concat(importance_rows, ignore_index=True).to_csv(
                output_dir / f"importance_{experiment}.csv",
                index=False,
            )
            pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)
            pd.DataFrame(fold_results).to_csv(output_dir / "fold_results.csv", index=False)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks",
        nargs="+",
        choices=["returns_compact", "returns", "non_return", "combined"],
        default=["returns"],
    )
    parser.add_argument(
        "--leaf-types",
        nargs="+",
        choices=["constant", "linear"],
        default=["constant", "linear"],
    )
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument(
        "--category-set",
        choices=["full", "cluster", "group"],
        default="full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "v3_lgbm_screen",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    table = run_screen(
        blocks_to_run=args.blocks,
        leaf_types=args.leaf_types,
        max_folds=args.max_folds,
        n_estimators=args.n_estimators,
        output_dir=args.output_dir,
        category_set=args.category_set,
    )
    print(table.sort_values("accuracy", ascending=False).to_string(index=False))

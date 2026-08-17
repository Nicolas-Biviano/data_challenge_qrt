"""H12: strongly regularized LightGBM on returns and conditional market states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from gpt.h11_conditional_allocation_response import (  # noqa: E402
    STATE_FEATURES,
    build_state_features,
)
from gpt.lightgbm_diagnostics import collect_structure  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h12_lgbm_conditional"
REFERENCE_V2 = (
    GPT_DIR
    / "outputs"
    / "h11_conditional_allocation_response"
    / "oof_baseline_raw.csv"
)
ITERATIONS = [25, 50, 100, 150]


def parameters(linear_tree: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "objective": "binary",
        "n_estimators": max(ITERATIONS),
        "learning_rate": 0.03,
        "max_depth": 3,
        "num_leaves": 8,
        "min_child_samples": 5000,
        "min_child_weight": 500.0,
        "min_split_gain": 0.1,
        "reg_alpha": 10.0,
        "reg_lambda": 100.0,
        "path_smooth": 100.0,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "feature_fraction_bynode": 0.8,
        "max_bin": 63,
        "cat_l2": 100.0,
        "cat_smooth": 100.0,
        "min_data_per_group": 500,
        "max_cat_threshold": 32,
        "force_col_wise": True,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 0,
    }
    if linear_tree:
        result["linear_tree"] = True
        result["linear_lambda"] = 100.0
    return result


def matrix(
    X: pd.DataFrame,
    numeric_columns: list[str],
    allocation_categories: pd.Index,
) -> pd.DataFrame:
    result = X[numeric_columns].astype("float32").copy()
    result["ALLOCATION"] = pd.Categorical(
        X.ALLOCATION.astype(str),
        categories=allocation_categories,
    )
    result["GROUP"] = pd.Categorical(X.GROUP.astype(str))
    return result


def run(output_dir: Path, max_folds: int = 2) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = build_state_features(ChallengeDataLoader.load_X_train())
    y = ChallengeDataLoader.load_y_train().target_binarized.astype(int)
    numeric = BASE_RETURNS + STATE_FEATURES
    unique_dates = X.TS.unique()
    splits = list(KFold(n_splits=8, shuffle=True, random_state=0).split(unique_dates))
    result_rows = []
    curve_rows = []
    structure_rows = []

    for leaf_type, is_linear in (("constant", False), ("linear", True)):
        oof_probability = pd.Series(np.nan, index=X.index)
        oof_fold = pd.Series(np.nan, index=X.index)
        for fold, (train_date_position, valid_date_position) in enumerate(splits, 1):
            if fold > max_folds:
                break
            train_mask = X.TS.isin(unique_dates[train_date_position])
            valid_mask = X.TS.isin(unique_dates[valid_date_position])
            allocation_categories = pd.Index(sorted(X.loc[train_mask, "ALLOCATION"].astype(str).unique()))
            train_matrix = matrix(X.loc[train_mask], numeric, allocation_categories)
            valid_matrix = matrix(X.loc[valid_mask], numeric, allocation_categories)
            model = lgb.LGBMClassifier(**parameters(is_linear))
            model.fit(
                train_matrix,
                y.loc[train_mask],
                categorical_feature=["ALLOCATION", "GROUP"],
            )
            rng = np.random.default_rng(fold)
            sample_position = rng.choice(
                len(train_matrix),
                min(100_000, len(train_matrix)),
                replace=False,
            )
            for iteration in ITERATIONS:
                train_probability = model.predict_proba(
                    train_matrix.iloc[sample_position],
                    num_iteration=iteration,
                )[:, 1]
                valid_probability = model.predict_proba(
                    valid_matrix,
                    num_iteration=iteration,
                )[:, 1]
                train_truth = y.loc[train_mask].iloc[sample_position].to_numpy()
                valid_truth = y.loc[valid_mask].to_numpy()
                train_accuracy = float(np.mean((train_probability > 0.5) == train_truth))
                valid_accuracy = float(np.mean((valid_probability > 0.5) == valid_truth))
                curve_rows.append(
                    {
                        "leaf_type": leaf_type,
                        "fold": fold,
                        "iteration": iteration,
                        "train_accuracy": train_accuracy,
                        "valid_accuracy": valid_accuracy,
                        "accuracy_gap": train_accuracy - valid_accuracy,
                        "train_auc": float(roc_auc_score(train_truth, train_probability)),
                        "valid_auc": float(roc_auc_score(valid_truth, valid_probability)),
                    }
                )
            final_probability = model.predict_proba(valid_matrix)[:, 1]
            oof_probability.loc[valid_mask] = final_probability
            oof_fold.loc[valid_mask] = fold
            structure_rows.append(
                {"leaf_type": leaf_type, "fold": fold, **collect_structure(model)}
            )
            pd.DataFrame(curve_rows).to_csv(output_dir / "training_curves.csv", index=False)
            pd.DataFrame(structure_rows).to_csv(output_dir / "structures.csv", index=False)
            print(
                json.dumps(
                    {
                        "leaf_type": leaf_type,
                        "fold": fold,
                        "accuracy": float(np.mean((final_probability > 0.5) == y.loc[valid_mask].to_numpy())),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        covered = oof_probability.notna()
        truth = y.loc[covered].to_numpy()
        correct = (oof_probability.loc[covered].to_numpy() > 0.5) == truth
        by_date = pd.Series(correct.astype(float), index=X.index[covered]).groupby(
            X.loc[covered, "TS"]
        ).mean()
        final_curves = pd.DataFrame(curve_rows)
        local_curve = final_curves[
            final_curves.leaf_type.eq(leaf_type)
            & final_curves.iteration.eq(max(ITERATIONS))
        ]
        local_structures = pd.DataFrame(structure_rows)
        local_structures = local_structures[local_structures.leaf_type.eq(leaf_type)]
        result_rows.append(
            {
                "leaf_type": leaf_type,
                "n_folds": max_folds,
                "accuracy": float(correct.mean()),
                "date_standard_error": float(by_date.std(ddof=1) / np.sqrt(len(by_date))),
                "auc": float(roc_auc_score(truth, oof_probability.loc[covered])),
                "mean_train_accuracy": float(local_curve.train_accuracy.mean()),
                "mean_accuracy_gap": float(local_curve.accuracy_gap.mean()),
                "actual_depth_mean": float(local_structures.actual_depth_mean.mean()),
                "actual_depth_max": int(local_structures.actual_depth_max.max()),
                "leaves_per_tree_mean": float(local_structures.leaves_per_tree_mean.mean()),
            }
        )
        pd.DataFrame(
            {
                "fold": oof_fold.loc[covered],
                "TS": X.loc[covered, "TS"],
                "ALLOCATION": X.loc[covered, "ALLOCATION"],
                "y_true_binarized": truth,
                "probability": oof_probability.loc[covered],
                "prediction": (oof_probability.loc[covered] > 0.5).astype(int),
                "is_correct": correct.astype(int),
            },
            index=X.index[covered],
        ).rename_axis("ROW_ID").to_csv(output_dir / f"oof_{leaf_type}.csv")
        pd.DataFrame(result_rows).to_csv(output_dir / "results.csv", index=False)

    reference = pd.read_csv(REFERENCE_V2, index_col="ROW_ID")
    baseline_accuracy = float(
        reference.loc[reference.fold.le(max_folds), "is_correct"].mean()
    )
    best = pd.DataFrame(result_rows).sort_values("accuracy", ascending=False).iloc[0]
    summary: dict[str, object] = {
        "configuration": parameters(best.leaf_type == "linear"),
        "baseline_v2_same_folds": baseline_accuracy,
        "best_leaf_type": best.leaf_type,
        "best_accuracy": float(best.accuracy),
        "gain_vs_v2": float(best.accuracy - baseline_accuracy),
        "full_eight_fold_authorized": bool(best.accuracy > baseline_accuracy),
        "results": result_rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments.output_dir, arguments.max_folds), indent=2, ensure_ascii=False))

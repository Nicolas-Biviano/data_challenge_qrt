"""H13: nested, paper-informed, heavily regularized Random Forest tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from gpt.h11_conditional_allocation_response import (  # noqa: E402
    PROFILE_FEATURES,
    STATE_FEATURES,
    allocation_style_profile,
    build_state_features,
)
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h13_paper_tuned_random_forest"
REFERENCE_V2 = (
    GPT_DIR / "outputs" / "h11_conditional_allocation_response" / "oof_baseline_raw.csv"
)
TREE_COUNTS = [100, 300, 600, 1000]


CONFIGS: list[dict[str, object]] = [
    {
        "name": "ultra_random",
        "max_depth": 3,
        "min_samples_leaf": 0.020,
        "max_features": 0.25,
        "max_samples": 0.40,
        "max_leaf_nodes": 8,
        "min_impurity_decrease": 0.0,
        "ccp_alpha": 0.0,
    },
    {
        "name": "very_strong",
        "max_depth": 4,
        "min_samples_leaf": 0.010,
        "max_features": 0.40,
        "max_samples": 0.50,
        "max_leaf_nodes": 12,
        "min_impurity_decrease": 1e-5,
        "ccp_alpha": 0.0,
    },
    {
        "name": "strong",
        "max_depth": 5,
        "min_samples_leaf": 0.005,
        "max_features": 0.60,
        "max_samples": 0.632,
        "max_leaf_nodes": 20,
        "min_impurity_decrease": 1e-5,
        "ccp_alpha": 0.0,
    },
    {
        "name": "pruned",
        "max_depth": 6,
        "min_samples_leaf": 0.0025,
        "max_features": 0.35,
        "max_samples": 0.50,
        "max_leaf_nodes": 24,
        "min_impurity_decrease": 2e-5,
        "ccp_alpha": 1e-5,
    },
]


ROW_FEATURES = BASE_RETURNS + STATE_FEATURES + [
    "ret1_market_residual",
    "ret1_group_residual",
]


def add_residuals(X: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()
    group_mean = result.market_ret1_mean + result.group_ret1_relative
    result["ret1_market_residual"] = result.RET_1 - result.market_ret1_mean
    result["ret1_group_residual"] = result.RET_1 - group_mean
    return result


def design_matrices(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    profiles = allocation_style_profile(train)
    profile_imputer = SimpleImputer(strategy="median")
    fitted_profiles = pd.DataFrame(
        profile_imputer.fit_transform(profiles),
        index=profiles.index.astype(str),
        columns=[f"profile_{name}" for name in PROFILE_FEATURES],
    )

    def assemble(frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame[ROW_FEATURES].astype("float32").copy()
        allocation_profiles = fitted_profiles.reindex(
            frame.ALLOCATION.astype(str).to_numpy()
        ).set_axis(frame.index)
        allocation_profiles = allocation_profiles.fillna(0.0).astype("float32")
        groups = pd.get_dummies(frame.GROUP.astype(str), prefix="GROUP", dtype="float32")
        groups = groups.reindex(columns=["GROUP_1", "GROUP_2", "GROUP_3", "GROUP_4"], fill_value=0.0)
        return pd.concat([numeric, allocation_profiles, groups], axis=1)

    train_design = assemble(train)
    valid_design = assemble(valid)
    imputer = SimpleImputer(strategy="median")
    train_matrix = imputer.fit_transform(train_design).astype("float32")
    valid_matrix = imputer.transform(valid_design).astype("float32")
    return train_matrix, valid_matrix, train_design.columns.tolist()


def make_forest(
    config: dict[str, object],
    n_estimators: int,
    random_state: int,
    warm_start: bool = False,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="log_loss",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state,
        warm_start=warm_start,
        class_weight=None,
        **{key: value for key, value in config.items() if key != "name"},
    )


def tree_structure(model: RandomForestClassifier) -> dict[str, float | int]:
    depths = np.array([tree.tree_.max_depth for tree in model.estimators_])
    leaves = np.array([tree.tree_.n_leaves for tree in model.estimators_])
    leaf_samples = []
    for tree in model.estimators_:
        is_leaf = tree.tree_.children_left == -1
        leaf_samples.extend(tree.tree_.n_node_samples[is_leaf].tolist())
    return {
        "actual_depth_mean": float(depths.mean()),
        "actual_depth_max": int(depths.max()),
        "leaves_mean": float(leaves.mean()),
        "leaves_max": int(leaves.max()),
        "leaf_samples_q05": float(np.quantile(leaf_samples, 0.05)),
        "leaf_samples_median": float(np.median(leaf_samples)),
        "leaf_samples_q95": float(np.quantile(leaf_samples, 0.95)),
    }


def probability_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(np.mean((probability > 0.5) == truth)),
        "auc": float(roc_auc_score(truth, probability)),
        "brier": float(brier_score_loss(truth, probability)),
    }


def run(
    output_dir: Path,
    max_outer_folds: int = 2,
    resume: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = add_residuals(build_state_features(ChallengeDataLoader.load_X_train()))
    y = ChallengeDataLoader.load_y_train().target_binarized.astype(int)
    unique_dates = X.TS.unique()
    outer_splits = list(KFold(n_splits=8, shuffle=True, random_state=0).split(unique_dates))
    def existing_rows(filename: str) -> list[dict[str, object]]:
        path = output_dir / filename
        if resume and path.exists():
            return pd.read_csv(path).to_dict(orient="records")
        return []

    inner_rows = existing_rows("inner_results.csv")
    outer_rows = existing_rows("outer_results.csv")
    curve_rows = existing_rows("tree_count_curves.csv")
    structure_rows = existing_rows("structures.csv")
    importance_rows = existing_rows("feature_importance.csv")
    oof_probability = pd.Series(np.nan, index=X.index)
    oof_fold = pd.Series(np.nan, index=X.index)
    oof_path = output_dir / "oof_predictions.csv"
    if resume and oof_path.exists():
        existing_oof = pd.read_csv(oof_path, index_col="ROW_ID")
        oof_probability.loc[existing_oof.index] = existing_oof.probability
        oof_fold.loc[existing_oof.index] = existing_oof.fold
    completed_folds = {int(row["outer_fold"]) for row in outer_rows}

    for outer_fold, (outer_train_position, outer_valid_position) in enumerate(outer_splits, 1):
        if outer_fold > max_outer_folds:
            break
        if outer_fold in completed_folds:
            continue
        outer_train_dates = unique_dates[outer_train_position]
        outer_valid_dates = unique_dates[outer_valid_position]
        inner_train_dates, inner_valid_dates = train_test_split(
            outer_train_dates,
            test_size=0.20,
            random_state=10_000 + outer_fold,
        )
        inner_train_mask = X.TS.isin(inner_train_dates)
        inner_valid_mask = X.TS.isin(inner_valid_dates)
        inner_train_matrix, inner_valid_matrix, feature_names = design_matrices(
            X.loc[inner_train_mask],
            X.loc[inner_valid_mask],
        )
        inner_truth_train = y.loc[inner_train_mask].to_numpy()
        inner_truth_valid = y.loc[inner_valid_mask].to_numpy()
        rng = np.random.default_rng(outer_fold)
        inner_train_sample = rng.choice(
            len(inner_truth_train),
            min(100_000, len(inner_truth_train)),
            replace=False,
        )
        local_inner_rows = []
        for config_position, config in enumerate(CONFIGS):
            model = make_forest(config, 300, 1_000 * outer_fold + config_position)
            model.fit(inner_train_matrix, inner_truth_train)
            train_probability = model.predict_proba(inner_train_matrix[inner_train_sample])[:, 1]
            valid_probability = model.predict_proba(inner_valid_matrix)[:, 1]
            train_metrics = probability_metrics(inner_truth_train[inner_train_sample], train_probability)
            valid_metrics = probability_metrics(inner_truth_valid, valid_probability)
            row = {
                "outer_fold": outer_fold,
                "config": config["name"],
                "n_estimators": 300,
                "oob_accuracy": float(model.oob_score_),
                "train_accuracy": train_metrics["accuracy"],
                "valid_accuracy": valid_metrics["accuracy"],
                "accuracy_gap": train_metrics["accuracy"] - valid_metrics["accuracy"],
                "valid_auc": valid_metrics["auc"],
                "valid_brier": valid_metrics["brier"],
            }
            local_inner_rows.append(row)
            inner_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        selected_name = max(local_inner_rows, key=lambda row: row["valid_accuracy"])["config"]
        selected = next(config for config in CONFIGS if config["name"] == selected_name)

        outer_train_mask = X.TS.isin(outer_train_dates)
        outer_valid_mask = X.TS.isin(outer_valid_dates)
        train_matrix, valid_matrix, feature_names = design_matrices(
            X.loc[outer_train_mask],
            X.loc[outer_valid_mask],
        )
        truth_train = y.loc[outer_train_mask].to_numpy()
        truth_valid = y.loc[outer_valid_mask].to_numpy()
        train_sample = rng.choice(len(truth_train), min(100_000, len(truth_train)), replace=False)
        forest = make_forest(selected, TREE_COUNTS[0], 20_000 + outer_fold, warm_start=True)
        for n_trees in TREE_COUNTS:
            forest.set_params(n_estimators=n_trees)
            forest.fit(train_matrix, truth_train)
            train_probability = forest.predict_proba(train_matrix[train_sample])[:, 1]
            valid_probability = forest.predict_proba(valid_matrix)[:, 1]
            train_metrics = probability_metrics(truth_train[train_sample], train_probability)
            valid_metrics = probability_metrics(truth_valid, valid_probability)
            curve_rows.append(
                {
                    "outer_fold": outer_fold,
                    "config": selected_name,
                    "n_estimators": n_trees,
                    "oob_accuracy": float(forest.oob_score_),
                    "train_accuracy": train_metrics["accuracy"],
                    "valid_accuracy": valid_metrics["accuracy"],
                    "accuracy_gap": train_metrics["accuracy"] - valid_metrics["accuracy"],
                    "valid_auc": valid_metrics["auc"],
                    "valid_brier": valid_metrics["brier"],
                }
            )
        final_probability = forest.predict_proba(valid_matrix)[:, 1]
        final_metrics = probability_metrics(truth_valid, final_probability)
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_config": selected_name,
                "n_estimators": TREE_COUNTS[-1],
                "oob_accuracy": float(forest.oob_score_),
                "train_accuracy": curve_rows[-1]["train_accuracy"],
                **{f"valid_{key}": value for key, value in final_metrics.items()},
                "accuracy_gap": curve_rows[-1]["accuracy_gap"],
                "positive_prediction_rate": float((final_probability > 0.5).mean()),
            }
        )
        structure_rows.append(
            {"outer_fold": outer_fold, "config": selected_name, **tree_structure(forest)}
        )
        importance_rows.extend(
            {
                "outer_fold": outer_fold,
                "feature": feature,
                "impurity_importance": float(importance),
            }
            for feature, importance in zip(feature_names, forest.feature_importances_)
        )
        oof_probability.loc[outer_valid_mask] = final_probability
        oof_fold.loc[outer_valid_mask] = outer_fold
        pd.DataFrame(inner_rows).to_csv(output_dir / "inner_results.csv", index=False)
        pd.DataFrame(outer_rows).to_csv(output_dir / "outer_results.csv", index=False)
        pd.DataFrame(curve_rows).to_csv(output_dir / "tree_count_curves.csv", index=False)
        pd.DataFrame(structure_rows).to_csv(output_dir / "structures.csv", index=False)
        pd.DataFrame(importance_rows).to_csv(output_dir / "feature_importance.csv", index=False)
        print(json.dumps(outer_rows[-1], ensure_ascii=False), flush=True)

    covered = oof_probability.notna()
    truth = y.loc[covered].to_numpy()
    correct = (oof_probability.loc[covered].to_numpy() > 0.5) == truth
    reference = pd.read_csv(REFERENCE_V2, index_col="ROW_ID")
    baseline = reference.loc[covered]
    baseline_correct = baseline.is_correct.to_numpy(int)
    baseline_probability = baseline.score.to_numpy(float)
    row_gain = correct.astype(int) - baseline_correct
    date_gain = pd.Series(row_gain, index=X.index[covered]).groupby(X.loc[covered, "TS"]).mean()
    gain = float(row_gain.mean())
    se = float(date_gain.std(ddof=1) / np.sqrt(len(date_gain)))
    summary: dict[str, object] = {
        "model": "nested paper-informed heavily regularized RandomForestClassifier",
        "n_outer_folds": max_outer_folds,
        "n_final_trees_fixed": TREE_COUNTS[-1],
        "baseline_v2_accuracy": float(baseline_correct.mean()),
        "rf_accuracy": float(correct.mean()),
        "rf_auc": float(roc_auc_score(truth, oof_probability.loc[covered])),
        "rf_brier": float(brier_score_loss(truth, oof_probability.loc[covered])),
        "rf_positive_prediction_rate": float(
            (oof_probability.loc[covered].to_numpy() > 0.5).mean()
        ),
        "v2_auc": float(roc_auc_score(truth, baseline_probability)),
        "v2_brier": float(brier_score_loss(truth, baseline_probability)),
        "v2_positive_prediction_rate": float((baseline_probability > 0.5).mean()),
        "target_positive_rate": float(truth.mean()),
        "gain_vs_v2": gain,
        "date_paired_standard_error": se,
        "ci95_low": gain - 1.96 * se,
        "ci95_high": gain + 1.96 * se,
        "folds_won": int(
            sum(row["valid_accuracy"] > float(
                baseline.loc[baseline.fold.eq(row["outer_fold"]), "is_correct"].mean()
            ) for row in outer_rows)
        ),
        "mean_train_valid_gap": float(pd.DataFrame(outer_rows).accuracy_gap.mean()),
        "selected_configs": [row["selected_config"] for row in outer_rows],
        "full_eight_fold_authorized": bool(
            gain > 0
            and pd.DataFrame(outer_rows).accuracy_gap.mean() < 0.01
            and sum(row["valid_accuracy"] > float(
                baseline.loc[baseline.fold.eq(row["outer_fold"]), "is_correct"].mean()
            ) for row in outer_rows) >= max(1, max_outer_folds // 2)
        ),
        "sources": [
            "https://arxiv.org/abs/1804.03515",
            "https://www.jmlr.org/papers/v21/19-905.html",
            "https://arxiv.org/abs/1705.05654",
            "https://doi.org/10.1023/A:1010933404324",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
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
    ).rename_axis("ROW_ID").to_csv(output_dir / "oof_predictions.csv")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-outer-folds", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            run(arguments.output_dir, arguments.max_outer_folds, arguments.resume),
            indent=2,
            ensure_ascii=False,
        )
    )

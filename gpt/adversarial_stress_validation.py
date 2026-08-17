"""Validation adversariale train-test et holdout V2 sur dates test-like."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from gpt.model import FixedEffectDesign  # noqa: E402
from gpt.model_v2 import MODEL_C, REFERENCE_OOF  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


SELECTED_VOLUMES = [
    "SIGNED_VOLUME_1",
    "SIGNED_VOLUME_2",
    "SIGNED_VOLUME_3",
    "SIGNED_VOLUME_5",
    "SIGNED_VOLUME_10",
    "SIGNED_VOLUME_15",
    "SIGNED_VOLUME_20",
]
PROFILE_COLUMNS = BASE_RETURNS + SELECTED_VOLUMES + ["MEDIAN_DAILY_TURNOVER"]
MODEL_FEATURES = ["ALLOCATION", "GROUP"] + BASE_RETURNS


def date_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume chaque date avec des statistiques uniquement issues de X."""
    grouped = frame.groupby("TS", observed=True)
    profile = grouped[PROFILE_COLUMNS].agg(["mean", "std"])
    profile.columns = [f"date_{stat}_{column}" for column, stat in profile.columns]
    missing = frame[["TS"] + SELECTED_VOLUMES].copy()
    missing[SELECTED_VOLUMES] = missing[SELECTED_VOLUMES].isna().astype("float32")
    missing_profile = missing.groupby("TS", observed=True)[SELECTED_VOLUMES].mean()
    missing_profile.columns = [f"date_missing_{column}" for column in SELECTED_VOLUMES]
    group_share = pd.crosstab(frame["TS"], frame["GROUP"], normalize="index")
    group_share.columns = [f"date_group_share_{value}" for value in group_share.columns]
    structural = pd.DataFrame(
        {
            "date_n_rows": grouped.size(),
            "date_n_allocations": grouped["ALLOCATION"].nunique(),
        }
    )
    return pd.concat(
        [profile, missing_profile, group_share, structural],
        axis=1,
    ).astype("float32")


def build_domain_matrices(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.Series, pd.Series, pd.Series]:
    """Construit blocs ligne/date avec codes categoriels communs."""
    train = X_train.copy()
    test = X_test.copy()
    train["__domain"] = 0
    test["__domain"] = 1
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    domain = combined.pop("__domain").astype("int8")
    groups = combined["TS"].astype(str)

    raw_numeric = [
        column
        for column in combined.columns
        if column not in {"TS", "ALLOCATION", "GROUP"}
    ]
    row = combined[raw_numeric].astype("float32")
    allocation_categories = pd.Categorical(combined["ALLOCATION"])
    row["ALLOCATION_CODE"] = allocation_categories.codes.astype("int32")
    row["GROUP_CODE"] = pd.Categorical(combined["GROUP"]).codes.astype("int32")

    profile = date_profile(combined)
    date = profile.reindex(groups).reset_index(drop=True)
    combined_matrix = pd.concat(
        [row.reset_index(drop=True), date.add_prefix("PROFILE_")],
        axis=1,
    )
    matrices = {
        "row": row.reset_index(drop=True),
        "date": date,
        "combined": combined_matrix,
    }
    source_row_id = pd.Series(
        np.concatenate([X_train.index.to_numpy(), X_test.index.to_numpy()]),
        name="source_row_id",
    )
    return matrices, domain, groups, source_row_id


def adversarial_model() -> lgb.LGBMClassifier:
    """Petit arbre regularise pour detecter un shift non lineaire."""
    return lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        n_estimators=200,
        max_depth=3,
        num_leaves=7,
        min_child_samples=1000,
        min_child_weight=1000.0,
        reg_alpha=5.0,
        reg_lambda=100.0,
        min_split_gain=0.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbosity=-1,
        n_jobs=-1,
        random_state=0,
    )


def run_adversarial_cv(
    matrices: dict[str, pd.DataFrame],
    domain: pd.Series,
    groups: pd.Series,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """OOF adversarial groupe par date pour chaque bloc de features."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    predictions = pd.DataFrame(index=domain.index)
    fold_rows = []
    importance_rows = []
    for block, matrix in matrices.items():
        block_prediction = pd.Series(np.nan, index=domain.index)
        categorical = (
            [matrix.columns.get_loc("ALLOCATION_CODE"), matrix.columns.get_loc("GROUP_CODE")]
            if block == "row"
            else [
                matrix.columns.get_loc("ALLOCATION_CODE"),
                matrix.columns.get_loc("GROUP_CODE"),
            ]
            if block == "combined"
            else []
        )
        block_importances = []
        for fold, (train_index, valid_index) in enumerate(
            splitter.split(matrix, domain, groups), start=1
        ):
            y_train = domain.iloc[train_index]
            counts = y_train.value_counts()
            weights = y_train.map(
                {
                    label: len(y_train) / (2.0 * count)
                    for label, count in counts.items()
                }
            )
            model = adversarial_model()
            model.fit(
                matrix.iloc[train_index],
                y_train,
                sample_weight=weights,
                categorical_feature=categorical,
            )
            probability = model.predict_proba(matrix.iloc[valid_index])[:, 1]
            block_prediction.iloc[valid_index] = probability
            fold_rows.append(
                {
                    "block": block,
                    "fold": fold,
                    "auc": roc_auc_score(domain.iloc[valid_index], probability),
                    "n_valid": len(valid_index),
                    "n_valid_test": int(domain.iloc[valid_index].sum()),
                }
            )
            block_importances.append(model.booster_.feature_importance("gain"))
        predictions[block] = block_prediction
        mean_importance = np.mean(block_importances, axis=0)
        importance_rows.extend(
            {
                "block": block,
                "feature": feature,
                "gain_importance": float(importance),
            }
            for feature, importance in zip(matrix.columns, mean_importance)
        )

    predictions["domain"] = domain
    predictions["TS"] = groups
    fold_table = pd.DataFrame(fold_rows)
    importance_table = pd.DataFrame(importance_rows).sort_values(
        ["block", "gain_importance"], ascending=[True, False]
    )
    fold_table.to_csv(output_dir / "adversarial_fold_metrics.csv", index=False)
    importance_table.to_csv(output_dir / "adversarial_importance.csv", index=False)
    return predictions, fold_table, importance_table


def v2_stress_holdout(
    X: pd.DataFrame,
    y: pd.DataFrame,
    valid_dates: pd.Index,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Reentraine V2 hors des dates test-like et predit ce holdout."""
    valid_mask = X["TS"].isin(valid_dates)
    train_mask = ~valid_mask
    design = FixedEffectDesign().fit(X.loc[train_mask, MODEL_FEATURES])
    train_matrix = design.transform(X.loc[train_mask, MODEL_FEATURES])
    valid_matrix = design.transform(X.loc[valid_mask, MODEL_FEATURES])
    model = LogisticRegression(
        C=MODEL_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=200,
        tol=1e-5,
        random_state=0,
    )
    model.fit(train_matrix, y.loc[train_mask, "target_binarized"])
    probability = model.predict_proba(valid_matrix)[:, 1]
    truth = y.loc[valid_mask, "target_binarized"].to_numpy(dtype="int8")
    rows = []
    train_positive_rate = float(y.loc[train_mask, "target_binarized"].mean())
    rate_matched_threshold = float(np.quantile(probability, 1.0 - train_positive_rate))
    thresholds = {
        "natural_0.5": 0.5,
        "fixed_0.502": 0.502,
        "fixed_0.504": 0.504,
        "fixed_0.506": 0.506,
        "match_train_positive_rate": rate_matched_threshold,
    }
    for name, threshold in thresholds.items():
        prediction = (probability > threshold).astype("int8")
        rows.append(
            {
                "threshold_name": name,
                "threshold": threshold,
                "accuracy": accuracy_score(truth, prediction),
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "prediction_positive_rate": float(prediction.mean()),
                "target_positive_rate": float(truth.mean()),
            }
        )
    prediction_table = pd.DataFrame(
        {
            "ROW_ID": X.index[valid_mask],
            "TS": X.loc[valid_mask, "TS"],
            "target": truth,
            "probability": probability,
        }
    ).set_index("ROW_ID")
    metadata = {
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "n_train_dates": int(X.loc[train_mask, "TS"].nunique()),
        "n_valid_dates": int(len(valid_dates)),
        "auc": float(roc_auc_score(truth, probability)),
    }
    return pd.DataFrame(rows), prediction_table, metadata


def run(output_dir: Path) -> dict[str, object]:
    """Execute adversarial CV, selection test-like et stress V2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train = ChallengeDataLoader.load_X_train()
    X_test = ChallengeDataLoader.load_X_test()
    y = ChallengeDataLoader.load_y_train()
    matrices, domain, groups, source_row_id = build_domain_matrices(X_train, X_test)
    adversarial, fold_table, importance = run_adversarial_cv(
        matrices, domain, groups, output_dir
    )
    adversarial["source_row_id"] = source_row_id
    adversarial.to_csv(output_dir / "adversarial_oof.csv", index=False)

    combined_auc = float(roc_auc_score(domain, adversarial["combined"]))
    train_domain = adversarial.loc[adversarial["domain"].eq(0)].copy()
    date_scores = (
        train_domain.groupby("TS", observed=True)["combined"]
        .agg(test_likeness="mean", score_std="std", n_rows="size")
        .sort_values("test_likeness", ascending=False)
    )
    date_scores.to_csv(output_dir / "train_date_test_likeness.csv")

    reference = pd.read_csv(REFERENCE_OOF, index_col="ROW_ID")
    oof_stress_rows = []
    for n_dates in (120, 240, 504):
        selected_dates = date_scores.head(n_dates).index
        selected = reference[reference["TS"].isin(selected_dates)]
        oof_stress_rows.append(
            {
                "n_dates": n_dates,
                "n_rows": len(selected),
                "accuracy": float(selected["is_correct"].mean()),
                "target_positive_rate": float(selected["y_true_binarized"].mean()),
                "prediction_positive_rate": float(selected["prediction"].mean()),
                "auc": float(
                    roc_auc_score(selected["y_true_binarized"], selected["score"])
                ),
            }
        )
    oof_stress = pd.DataFrame(oof_stress_rows)
    oof_stress.to_csv(output_dir / "v2_oof_by_test_likeness.csv", index=False)

    stress_dates = date_scores.head(120).index
    threshold_table, stress_predictions, stress_metadata = v2_stress_holdout(
        X_train, y, stress_dates
    )
    threshold_table.to_csv(output_dir / "stress_thresholds.csv", index=False)
    stress_predictions.to_csv(output_dir / "stress_predictions.csv")

    summary = {
        "adversarial_auc": {
            block: float(roc_auc_score(domain, adversarial[block]))
            for block in matrices
        },
        "adversarial_fold_auc": fold_table.groupby("block")["auc"].agg(
            ["mean", "std", "min", "max"]
        ).to_dict("index"),
        "stress_holdout": stress_metadata,
        "top_importance_combined": importance[
            importance["block"].eq("combined")
        ].head(15).to_dict("records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print("ADVERSARIAL AUC")
    print(pd.Series(summary["adversarial_auc"]).to_string())
    print("\nFOLD AUC")
    print(fold_table.groupby("block")["auc"].agg(["mean", "std", "min", "max"]).to_string())
    print("\nTOP COMBINED IMPORTANCE")
    print(
        importance[importance["block"].eq("combined")]
        .head(15)
        .to_string(index=False)
    )
    print("\nV2 OOF ON TEST-LIKE DATES")
    print(oof_stress.to_string(index=False))
    print("\nFRESH V2 STRESS HOLDOUT")
    print(pd.Series(stress_metadata).to_string())
    print(threshold_table.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "adversarial_stress_validation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_dir)

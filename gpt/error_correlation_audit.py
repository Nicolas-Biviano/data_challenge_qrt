"""Correlation des erreurs OOF et audit rapide du shift train-test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import ChallengeDataLoader  # noqa: E402


OOF_FILES = {
    "ridge_v1": "v1_recheck/oof_predictions.csv",
    "logistic_v2": "classification_full/oof_logistic_C_0.003.csv",
    "lgbm_linear": "v3_lgbm_linear_full8/oof_lgbm_linear_returns_compact_full.csv",
    "volume_specialist": "volume_specialist_full8/oof_observed_returns_C_0.003.csv",
    "sv_quantile_shape": "quantile_shape_full8/oof_sv_positive_shape_C_0.003.csv",
}


def load_predictions(outputs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligne predictions et erreurs binaires sur ROW_ID."""
    predictions = {}
    truth = None
    for name, relative_path in OOF_FILES.items():
        frame = pd.read_csv(outputs_root / relative_path, index_col="ROW_ID")
        predictions[name] = frame["prediction"].astype("int8")
        if truth is None:
            truth = frame["y_true_binarized"].astype("int8")
    prediction_table = pd.DataFrame(predictions).dropna().astype("int8")
    aligned_truth = truth.reindex(prediction_table.index).astype("int8")
    error_table = prediction_table.ne(aligned_truth, axis="index").astype("int8")
    return prediction_table, error_table


def pairwise_tables(
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcule correlation, desaccord, double faute et oracle par paire."""
    rows = []
    models = list(predictions)
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            left_error = errors[left].astype(bool)
            right_error = errors[right].astype(bool)
            rows.append(
                {
                    "model_left": left,
                    "model_right": right,
                    "error_correlation": float(errors[left].corr(errors[right])),
                    "prediction_disagreement": float(
                        predictions[left].ne(predictions[right]).mean()
                    ),
                    "double_fault_rate": float((left_error & right_error).mean()),
                    "oracle_pair_accuracy": float((~(left_error & right_error)).mean()),
                    "right_correct_when_left_wrong": float(
                        (~right_error[left_error]).mean()
                    ),
                    "left_correct_when_right_wrong": float(
                        (~left_error[right_error]).mean()
                    ),
                }
            )
    return errors.corr(), pd.DataFrame(rows)


def total_variation(train: pd.Series, test: pd.Series) -> dict[str, float | int]:
    """Distance de distribution et categories absentes entre train et test."""
    train_rate = train.value_counts(normalize=True, dropna=False)
    test_rate = test.value_counts(normalize=True, dropna=False)
    categories = train_rate.index.union(test_rate.index)
    train_rate = train_rate.reindex(categories, fill_value=0.0)
    test_rate = test_rate.reindex(categories, fill_value=0.0)
    return {
        "total_variation": float(0.5 * (train_rate - test_rate).abs().sum()),
        "train_categories": int(train.nunique(dropna=False)),
        "test_categories": int(test.nunique(dropna=False)),
        "unseen_test_categories": int(len(set(test.dropna()) - set(train.dropna()))),
    }


def numerical_shift(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Resume missingness, moyenne standardisee et KS des colonnes numeriques."""
    rows = []
    rng = np.random.default_rng(0)
    numeric_columns = X_train.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        train = X_train[column].dropna().to_numpy(float)
        test = X_test[column].dropna().to_numpy(float)
        if len(train) == 0 or len(test) == 0:
            continue
        if len(train) > 100_000:
            train = rng.choice(train, 100_000, replace=False)
        pooled_std = np.sqrt(0.5 * (np.var(train) + np.var(test)))
        standardized_mean_difference = (
            float((np.mean(test) - np.mean(train)) / pooled_std)
            if pooled_std > 0.0
            else 0.0
        )
        rows.append(
            {
                "feature": column,
                "train_missing_rate": float(X_train[column].isna().mean()),
                "test_missing_rate": float(X_test[column].isna().mean()),
                "missing_rate_shift": float(
                    X_test[column].isna().mean() - X_train[column].isna().mean()
                ),
                "train_mean": float(np.mean(train)),
                "test_mean": float(np.mean(test)),
                "standardized_mean_difference": standardized_mean_difference,
                "ks_statistic": float(ks_2samp(train, test).statistic),
            }
        )
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)


def run_audit(outputs_root: Path, output_dir: Path) -> dict[str, object]:
    """Produit les tableaux de correlation d'erreurs et de shift."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, errors = load_predictions(outputs_root)
    error_correlation, pairs = pairwise_tables(predictions, errors)
    model_summary = pd.DataFrame(
        {
            "model": predictions.columns,
            "accuracy": [1.0 - float(errors[column].mean()) for column in predictions],
            "prediction_positive_rate": [
                float(predictions[column].mean()) for column in predictions
            ],
        }
    ).sort_values("accuracy", ascending=False)

    X_train = ChallengeDataLoader.load_X_train()
    X_test = ChallengeDataLoader.load_X_test()
    feature_shift = numerical_shift(X_train, X_test)
    category_shift = {
        column: total_variation(X_train[column], X_test[column])
        for column in ["ALLOCATION", "GROUP"]
    }
    train_rows_per_date = X_train.groupby("TS", observed=True).size()
    test_rows_per_date = X_test.groupby("TS", observed=True).size()
    category_shift["TS_rows"] = {
        "train_n_dates": int(len(train_rows_per_date)),
        "test_n_dates": int(len(test_rows_per_date)),
        "train_mean_rows": float(train_rows_per_date.mean()),
        "test_mean_rows": float(test_rows_per_date.mean()),
        "ks_statistic": float(
            ks_2samp(train_rows_per_date, test_rows_per_date).statistic
        ),
    }
    v2_test = pd.read_csv(outputs_root / "v2/test_predictions.csv", index_col="ROW_ID")
    prediction_shift = {
        "v2_oof_positive_rate": float(predictions["logistic_v2"].mean()),
        "v2_test_positive_rate": float(v2_test["prediction"].mean()),
        "v2_test_probability_mean": float(v2_test["probability_positive"].mean()),
        "v2_test_probability_std": float(v2_test["probability_positive"].std()),
    }

    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    error_correlation.to_csv(output_dir / "error_correlation.csv")
    pairs.to_csv(output_dir / "pairwise_complementarity.csv", index=False)
    feature_shift.to_csv(output_dir / "feature_shift.csv", index=False)
    summary = {
        "n_aligned_rows": int(len(predictions)),
        "category_shift": category_shift,
        "prediction_shift": prediction_shift,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print("MODEL SUMMARY")
    print(model_summary.to_string(index=False))
    print("\nERROR CORRELATION")
    print(error_correlation.round(4).to_string())
    print("\nCOMPLEMENTARITY VS V2")
    print(
        pairs[
            pairs["model_left"].eq("logistic_v2")
            | pairs["model_right"].eq("logistic_v2")
        ].to_string(index=False)
    )
    print("\nTOP NUMERICAL SHIFTS")
    print(feature_shift.head(12).to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "error_correlation_audit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_audit(args.outputs_root, args.output_dir)

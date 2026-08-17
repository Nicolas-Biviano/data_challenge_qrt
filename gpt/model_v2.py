"""Modele GPT V2 final: classification logistique sparse regularisee.

La V2 conserve la representation parcimonieuse de la V1 mais optimise
directement le signe de la cible. Aucun target encoding ni ordre de date n'est
utilise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.model import MODEL_FEATURES, FixedEffectDesign  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


MODEL_C = 0.003
REFERENCE_OOF = (
    Path(__file__).resolve().parent
    / "outputs"
    / "classification_full"
    / "oof_logistic_C_0.003.csv"
)


def summarize_reference_oof(path: Path) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Resume les predictions OOF deja calculees sur les huit folds fixes."""
    oof = pd.read_csv(path, index_col="ROW_ID")
    correct = oof["is_correct"].astype(float)
    by_ts = correct.groupby(oof["TS"]).mean()
    by_allocation = correct.groupby(oof["ALLOCATION"]).mean()
    accuracy = float(correct.mean())
    summary: dict[str, float | int] = {
        "accuracy": accuracy,
        "ts_standard_error": float(by_ts.std() / np.sqrt(len(by_ts))),
        "ts_penalized_score": float(accuracy - by_ts.std() / np.sqrt(len(by_ts))),
        "allocation_standard_error": float(
            by_allocation.std() / np.sqrt(len(by_allocation))
        ),
        "n_splits": int(oof["fold"].nunique()),
        "n_oof_rows": int(len(oof)),
    }
    fold_metrics = (
        oof.groupby("fold", observed=True)["is_correct"]
        .agg(accuracy="mean", n_valid="size")
        .reset_index()
    )
    return summary, fold_metrics


def fit_full_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    FixedEffectDesign,
    LogisticRegression,
    pd.DataFrame,
]:
    """Ajuste la V2 sur tout le train et predit les probabilites du test."""
    design = FixedEffectDesign().fit(X_train[MODEL_FEATURES])
    train_matrix = design.transform(X_train[MODEL_FEATURES])
    test_matrix = design.transform(X_test[MODEL_FEATURES])
    model = LogisticRegression(
        C=MODEL_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=150,
        tol=1e-5,
        random_state=0,
    )
    model.fit(train_matrix, y_train["target_binarized"])
    probabilities = model.predict_proba(test_matrix)[:, 1]
    predictions = (probabilities > 0.5).astype("int8")
    feature_names = design.get_feature_names()
    feature_mean = np.asarray(train_matrix.mean(axis=0)).ravel()
    feature_second_moment = np.asarray(train_matrix.power(2).mean(axis=0)).ravel()
    feature_std = np.sqrt(np.maximum(feature_second_moment - feature_mean**2, 0.0))
    coefficients = model.coef_.ravel()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "feature_std": feature_std,
            "contribution_std": np.abs(coefficients) * feature_std,
        }
    ).sort_values("contribution_std", ascending=False)
    return predictions, probabilities, design, model, importance


def run(output_dir: Path, reference_oof: Path = REFERENCE_OOF) -> dict[str, object]:
    """Produit la soumission, les scores test et le resume final."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train = ChallengeDataLoader.load_X_train()
    y_train = ChallengeDataLoader.load_y_train()
    X_test = ChallengeDataLoader.load_X_test()
    sample = ChallengeDataLoader.load_sample_submission()

    predictions, probabilities, _, model, importance = fit_full_and_predict(
        X_train,
        y_train,
        X_test,
    )
    submission = sample.copy()
    submission.iloc[:, 0] = predictions
    if not submission.index.equals(sample.index):
        raise ValueError("L'ordre des ROW_ID de la soumission a change")
    if submission.isna().any().any():
        raise ValueError("La soumission contient des valeurs manquantes")
    if not set(submission.iloc[:, 0].unique()).issubset({0, 1}):
        raise ValueError("La soumission doit etre binaire")

    submission.to_csv(output_dir / "submission.csv")
    pd.DataFrame(
        {"prediction": predictions, "probability_positive": probabilities},
        index=X_test.index,
    ).rename_axis("ROW_ID").to_csv(output_dir / "test_predictions.csv")
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    cv_summary, fold_metrics = summarize_reference_oof(reference_oof)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary: dict[str, object] = {
        "model": "Logistic regression with allocation/group fixed effects and allocation x RET_1 slopes",
        "C": MODEL_C,
        "features": MODEL_FEATURES,
        "target_encoding": False,
        "date_order_used": False,
        "cv": cv_summary,
        "test": {
            "n_rows": int(len(predictions)),
            "positive_rate": float(predictions.mean()),
            "probability_mean": float(probabilities.mean()),
            "probability_std": float(probabilities.std()),
        },
        "fit": {
            "n_iter": int(np.max(model.n_iter_)),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "v2",
    )
    parser.add_argument("--reference-oof", type=Path, default=REFERENCE_OOF)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.output_dir, args.reference_oof), indent=2, ensure_ascii=False))

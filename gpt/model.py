"""Premiere tentative de modelisation GPT pour le challenge QRT.

Le script reutilise volontairement ``src.cross_validation.run_cv``. Le modele
est une regression Ridge sur la cible continue avec :

* des effets fixes pour ALLOCATION et GROUP ;
* une selection parcimonieuse de rendements retardes ;
* une pente de RET_1 propre a chaque allocation.

La prediction finale est le signe de la cible continue predite.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_validation import CVconfig, ModelConfig, run_cv  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


NUMERIC_FEATURES = [
    "RET_1",
    "RET_2",
    "RET_3",
    "RET_4",
    "RET_7",
    "RET_8",
    "RET_9",
    "RET_18",
]
CATEGORICAL_FEATURES = ["ALLOCATION", "GROUP"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class FixedEffectDesign:
    """Construit la matrice sparse sans utiliser la cible."""

    interaction_scale: float = 1.0

    def __post_init__(self) -> None:
        self.numeric_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
        self.category_encoder = OneHotEncoder(handle_unknown="ignore")
        self.allocation_encoder = OneHotEncoder(handle_unknown="ignore")

    def fit(self, X: pd.DataFrame) -> "FixedEffectDesign":
        self.numeric_pipeline.fit(X[NUMERIC_FEATURES])
        self.category_encoder.fit(X[CATEGORICAL_FEATURES])
        self.allocation_encoder.fit(X[["ALLOCATION"]])
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        numeric = self.numeric_pipeline.transform(X[NUMERIC_FEATURES])
        categories = self.category_encoder.transform(X[CATEGORICAL_FEATURES])
        allocation = self.allocation_encoder.transform(X[["ALLOCATION"]])

        # RET_1 est la premiere colonne numerique standardisee. Multiplier le
        # one-hot d'allocation par RET_1 revient a estimer une pente locale,
        # avec shrinkage Ridge vers la pente commune.
        allocation_ret1 = allocation.multiply(numeric[:, 0, None])
        allocation_ret1 *= self.interaction_scale

        return sparse.hstack(
            [sparse.csr_matrix(numeric), categories, allocation_ret1],
            format="csr",
        )

    def get_feature_names(self) -> list[str]:
        """Retourne les noms de colonnes dans le meme ordre que transform."""
        category_names = self.category_encoder.get_feature_names_out(
            CATEGORICAL_FEATURES
        )
        allocation_names = self.allocation_encoder.get_feature_names_out(
            ["ALLOCATION"]
        )
        interaction_names = [
            f"{name} x RET_1" for name in allocation_names
        ]
        return (
            NUMERIC_FEATURES
            + list(category_names)
            + interaction_names
        )


def preprocess_fold(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Callback compatible avec le framework CV du projet."""
    design = FixedEffectDesign().fit(X_train)
    return design.transform(X_train), design.transform(X_valid)


def penalized_score(
    oof: pd.DataFrame,
    grouper: str,
    penalty: float = 1.0,
) -> dict[str, float | int]:
    """Reproduit CVresults.score tout en retournant ses composantes."""
    accuracy_by_group = oof.groupby(grouper, observed=True)["is_correct"].mean()
    accuracy = float(oof["is_correct"].mean())
    standard_error = float(accuracy_by_group.std() / np.sqrt(len(accuracy_by_group)))
    return {
        "accuracy": accuracy,
        "standard_error": standard_error,
        "penalty": penalty,
        "score": accuracy - penalty * standard_error,
        "n_groups": int(len(accuracy_by_group)),
    }


def summarize_cv(output: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    """Cree le resume JSON et la table de metriques par fold."""
    fold_rows = []
    for fold in output.fold_results:
        fold_rows.append(
            {
                "fold": fold.fold_id,
                "n_train": int(np.asarray(fold.train_index).sum()),
                "n_valid": int(np.asarray(fold.valid_index).sum()),
                "accuracy": float(fold.valid_metrics["accuracy"]),
                "balanced_accuracy": float(fold.valid_metrics["balanced_accuracy"]),
                "mcc": float(fold.valid_metrics["mcc"]),
                "mse": float(fold.valid_metrics["mse"]),
                "mae": float(fold.valid_metrics["mae"]),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = {
        "model": "Ridge regression with allocation fixed effects and allocation x RET_1 slopes",
        "alpha": 100.0,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_splits": int(output.n_folds),
        "scores": {
            grouper.lower(): penalized_score(output.oof_results, grouper)
            for grouper in ("fold", "TS", "ALLOCATION")
        },
    }
    return summary, fold_metrics


def fit_full_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Ajuste le meme modele sur tout le train et predit le test."""
    design = FixedEffectDesign().fit(X_train[MODEL_FEATURES])
    X_train_ready = design.transform(X_train[MODEL_FEATURES])
    X_test_ready = design.transform(X_test[MODEL_FEATURES])

    model = Ridge(alpha=100.0, solver="lsqr")
    model.fit(X_train_ready, y_train["target"])
    scores = model.predict(X_test_ready)
    predictions = (scores > 0.0).astype(np.int8)
    return predictions, scores


def run(n_splits: int, random_state: int, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = ChallengeDataLoader.load_X_train()
    y_train = ChallengeDataLoader.load_y_train()

    model_config = ModelConfig(
        model=Ridge(alpha=100.0, solver="lsqr"),
        regression=True,
        features=MODEL_FEATURES,
        preprocessing=preprocess_fold,
    )
    cv_config = CVconfig(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
        verbose=False,
    )
    cv_output = run_cv(X_train, y_train, model_config, cv_config)
    summary, fold_metrics = summarize_cv(cv_output)

    # Comparaison transparente au signal naif signe(RET_1), sur exactement les
    # memes observations et folds.
    baseline = cv_output.oof_results.copy()
    baseline["prediction"] = (X_train["RET_1"] > 0.0).astype(np.int8)
    baseline["is_correct"] = (
        baseline["prediction"] == baseline["y_true_binarized"]
    ).astype(np.int8)
    summary["ret1_sign_baseline"] = {
        grouper.lower(): penalized_score(baseline, grouper)
        for grouper in ("fold", "TS", "ALLOCATION")
    }

    oof_export = cv_output.oof_results.copy()
    oof_export["ret1_sign_prediction"] = baseline["prediction"]
    oof_export["ret1_sign_is_correct"] = baseline["is_correct"]
    oof_export.index.name = "ROW_ID"
    oof_export.to_csv(output_dir / "oof_predictions.csv")

    X_test = ChallengeDataLoader.load_X_test()
    predictions, raw_scores = fit_full_and_predict(X_train, y_train, X_test)

    sample = ChallengeDataLoader.load_sample_submission()
    submission = sample.copy()
    submission.iloc[:, 0] = predictions
    submission.to_csv(output_dir / "submission.csv")
    pd.DataFrame(
        {"prediction": predictions, "raw_score": raw_scores},
        index=X_test.index,
    ).rename_axis("ROW_ID").to_csv(output_dir / "test_predictions.csv")
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    summary["test"] = {
        "n_rows": int(len(predictions)),
        "positive_rate": float(predictions.mean()),
        "raw_score_mean": float(raw_scores.mean()),
        "raw_score_std": float(raw_scores.std()),
    }
    with (output_dir / "cv_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.n_splits, args.random_state, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))

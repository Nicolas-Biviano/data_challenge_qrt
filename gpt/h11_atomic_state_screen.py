"""Exploratory atomic screen of one allocation-by-state interaction at a time."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from gpt.h11_conditional_allocation_response import (  # noqa: E402
    CATEGORICAL,
    MODEL_C,
    STATE_FEATURES,
    build_state_features,
    make_preprocessor,
    paired_uncertainty,
)
from src.cross_validation import CVconfig, ModelConfig, run_cv  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h11_atomic_state_screen"
REFERENCE_OOF = (
    GPT_DIR
    / "outputs"
    / "h11_conditional_allocation_response"
    / "oof_baseline_raw.csv"
)


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = build_state_features(ChallengeDataLoader.load_X_train())
    y = ChallengeDataLoader.load_y_train()
    baseline = pd.read_csv(REFERENCE_OOF, index_col="ROW_ID")
    baseline.index = baseline.index.astype(X.index.dtype)
    baseline = baseline.loc[X.index]
    result_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for state in STATE_FEATURES:
        output = run_cv(
            X,
            y,
            ModelConfig(
                model=LogisticRegression(
                    C=MODEL_C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=250,
                    tol=1e-5,
                    random_state=0,
                ),
                regression=False,
                features=CATEGORICAL + BASE_RETURNS + [state],
                preprocessing=make_preprocessor([state], 0.0, 0.20, 0.0),
            ),
            CVconfig(n_splits=8, shuffle=True, random_state=0, verbose=False),
        )
        oof = output.oof_results
        uncertainty = paired_uncertainty(baseline, oof)
        result_rows.append(
            {
                "state": state,
                "accuracy": float(oof.is_correct.mean()),
                "positive_prediction_rate": float(oof.prediction.mean()),
                **uncertainty,
            }
        )
        for fold in output.fold_results:
            baseline_fold = baseline.loc[baseline.fold.eq(fold.fold_id), "is_correct"].mean()
            fold_rows.append(
                {
                    "state": state,
                    "fold": fold.fold_id,
                    "train_accuracy": float(fold.train_metrics["accuracy"]),
                    "accuracy": float(fold.valid_metrics["accuracy"]),
                    "gain_vs_baseline": float(fold.valid_metrics["accuracy"] - baseline_fold),
                    "train_auc": float(fold.train_metrics["auc"]),
                    "auc": float(fold.valid_metrics["auc"]),
                }
            )
        pd.DataFrame(result_rows).to_csv(output_dir / "results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
        print(json.dumps(result_rows[-1], ensure_ascii=False))

    results = pd.DataFrame(result_rows).sort_values("gain_accuracy", ascending=False)
    stable = results[results.folds_won.ge(5) & results.gain_accuracy.gt(0)]
    summary: dict[str, object] = {
        "purpose": "exploratory atomic screen; not an unbiased selected-model score",
        "allocation_interaction_scale": 0.20,
        "n_states": len(STATE_FEATURES),
        "best_state": results.iloc[0].state,
        "stable_positive_states": stable.state.tolist(),
        "results": results.to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))

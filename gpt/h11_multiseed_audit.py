"""Repeated date-fold audit of frozen V2 and H11 allocation responses."""

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
)
from src.cross_validation import CVconfig, ModelConfig, run_cv  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h11_multiseed_audit"
SEEDS = [0, 1, 2, 3, 4]


def model_config(candidate: bool) -> ModelConfig:
    states = STATE_FEATURES if candidate else []
    return ModelConfig(
        model=LogisticRegression(
            C=MODEL_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=250,
            tol=1e-5,
            random_state=0,
        ),
        regression=False,
        features=CATEGORICAL + BASE_RETURNS + states,
        preprocessing=make_preprocessor(
            states,
            group_scale=0.0,
            allocation_scale=0.20 if candidate else 0.0,
            style_scale=0.0,
        ),
    )


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = build_state_features(ChallengeDataLoader.load_X_train())
    y = ChallengeDataLoader.load_y_train()
    seed_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    repeated_gain: list[pd.DataFrame] = []

    for seed in SEEDS:
        outputs = {}
        for name, candidate in (("V2", False), ("H11", True)):
            outputs[name] = run_cv(
                X,
                y,
                model_config(candidate),
                CVconfig(n_splits=8, shuffle=True, random_state=seed, verbose=False),
            )
            for fold in outputs[name].fold_results:
                fold_rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "fold": fold.fold_id,
                        "train_accuracy": float(fold.train_metrics["accuracy"]),
                        "accuracy": float(fold.valid_metrics["accuracy"]),
                        "train_auc": float(fold.train_metrics["auc"]),
                        "auc": float(fold.valid_metrics["auc"]),
                    }
                )
        baseline = outputs["V2"].oof_results
        challenger = outputs["H11"].oof_results
        gain = challenger.is_correct.astype(float) - baseline.is_correct.astype(float)
        by_date = gain.groupby(baseline.TS).mean()
        se = float(by_date.std(ddof=1) / np.sqrt(len(by_date)))
        seed_gain = float(gain.mean())
        v2_accuracy = float(baseline.is_correct.mean())
        h11_accuracy = float(challenger.is_correct.mean())
        seed_rows.append(
            {
                "seed": seed,
                "v2_accuracy": v2_accuracy,
                "h11_accuracy": h11_accuracy,
                "gain_accuracy": seed_gain,
                "date_paired_standard_error": se,
                "ci95_low": seed_gain - 1.96 * se,
                "ci95_high": seed_gain + 1.96 * se,
                "folds_won": int(
                    (
                        challenger.groupby("fold").is_correct.mean()
                        - baseline.groupby("fold").is_correct.mean()
                    ).gt(0).sum()
                ),
            }
        )
        repeated_gain.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "TS": baseline.TS,
                    "row_gain": gain,
                },
                index=X.index,
            )
        )
        pd.DataFrame(seed_rows).to_csv(output_dir / "seed_results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output_dir / "fold_results.csv", index=False)
        print(json.dumps(seed_rows[-1], ensure_ascii=False))

    gains = pd.concat(repeated_gain)
    row_average = gains.groupby(gains.index).row_gain.mean()
    date_average = row_average.groupby(X.TS).mean()
    overall_gain = float(row_average.mean())
    overall_se = float(date_average.std(ddof=1) / np.sqrt(len(date_average)))
    seed_results = pd.DataFrame(seed_rows)
    folds = pd.DataFrame(fold_rows)
    fold_gain = folds.pivot_table(
        index=["seed", "fold"],
        columns="model",
        values="accuracy",
    )
    fold_gain["gain"] = fold_gain.H11 - fold_gain.V2
    summary: dict[str, object] = {
        "seeds": SEEDS,
        "models_frozen": True,
        "mean_v2_accuracy": float(seed_results.v2_accuracy.mean()),
        "mean_h11_accuracy": float(seed_results.h11_accuracy.mean()),
        "mean_gain_across_seeds": float(seed_results.gain_accuracy.mean()),
        "gain_from_row_averaged_repetitions": overall_gain,
        "date_paired_standard_error": overall_se,
        "ci95_low": overall_gain - 1.96 * overall_se,
        "ci95_high": overall_gain + 1.96 * overall_se,
        "positive_seeds": int(seed_results.gain_accuracy.gt(0).sum()),
        "total_folds_won": int(fold_gain.gain.gt(0).sum()),
        "total_folds": int(len(fold_gain)),
        "robustness_gate": bool(
            seed_results.gain_accuracy.gt(0).sum() >= 4
            and fold_gain.gain.gt(0).mean() >= 0.60
        ),
        "seed_results": seed_results.to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))

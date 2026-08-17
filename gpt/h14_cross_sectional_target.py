"""H14: strongly shrunk Ridge models with cross-sectionally normalized targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.model import FixedEffectDesign, MODEL_FEATURES  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h14_cross_sectional_target"
TARGET_NAMES = [
    "raw",
    "market_50",
    "market_100",
    "hierarchical_50",
    "hierarchical_100",
    "date_zscore",
    "date_zscore_clip3",
]


def build_targets(X: pd.DataFrame, target: pd.Series) -> dict[str, np.ndarray]:
    """Create target variants from training labels only."""
    work = pd.DataFrame(
        {
            "TS": X.TS.to_numpy(),
            "GROUP": X.GROUP.to_numpy(),
            "target": target.to_numpy(float),
        },
        index=X.index,
    )
    market_mean = work.groupby("TS", observed=True).target.transform("mean")
    market_std = work.groupby("TS", observed=True).target.transform("std")
    group_mean = work.groupby(["TS", "GROUP"], observed=True).target.transform("mean")
    positive_std = market_std[market_std > 0]
    floor = max(float(positive_std.median()) * 0.05, np.finfo(float).eps)
    safe_std = market_std.clip(lower=floor)
    zscore = (work.target - market_mean) / safe_std
    return {
        "raw": work.target.to_numpy(),
        "market_50": (work.target - 0.50 * market_mean).to_numpy(),
        "market_100": (work.target - market_mean).to_numpy(),
        "hierarchical_50": (
            work.target - market_mean - 0.50 * (group_mean - market_mean)
        ).to_numpy(),
        "hierarchical_100": (work.target - group_mean).to_numpy(),
        "date_zscore": zscore.to_numpy(),
        "date_zscore_clip3": zscore.clip(-3.0, 3.0).to_numpy(),
    }


def probability_from_score(score: np.ndarray) -> np.ndarray:
    """Monotone scale-free mapping used only for a Brier diagnostic."""
    scale = max(float(np.std(score)), np.finfo(float).eps)
    clipped = np.clip(score / scale, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def paired_summary(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
) -> dict[str, float | int]:
    gain_by_row = challenger.is_correct.to_numpy(float) - baseline.is_correct.to_numpy(float)
    gain_by_date = pd.Series(gain_by_row, index=baseline.TS).groupby(level=0).mean()
    gain_by_fold = pd.Series(gain_by_row, index=baseline.fold).groupby(level=0).mean()
    gain = float(gain_by_row.mean())
    se = float(gain_by_date.std(ddof=1) / np.sqrt(len(gain_by_date)))
    return {
        "gain_accuracy": gain,
        "date_paired_standard_error": se,
        "ci95_low": gain - 1.96 * se,
        "ci95_high": gain + 1.96 * se,
        "folds_won": int((gain_by_fold > 0).sum()),
    }


def run(output_dir: Path, max_folds: int = 4, random_state: int = 0) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    y_frame = ChallengeDataLoader.load_y_train()
    y_continuous = y_frame.target.astype(float)
    y_binary = y_frame.target_binarized.astype(int)
    unique_dates = X.TS.unique()
    splits = list(KFold(n_splits=8, shuffle=True, random_state=random_state).split(unique_dates))

    fold_rows: list[dict[str, object]] = []
    oof_rows: dict[str, list[pd.DataFrame]] = {name: [] for name in TARGET_NAMES}

    for fold, (train_position, valid_position) in enumerate(splits, 1):
        if fold > max_folds:
            break
        train_mask = X.TS.isin(unique_dates[train_position])
        valid_mask = X.TS.isin(unique_dates[valid_position])
        train = X.loc[train_mask]
        valid = X.loc[valid_mask]
        design = FixedEffectDesign().fit(train[MODEL_FEATURES])
        train_matrix = design.transform(train[MODEL_FEATURES])
        valid_matrix = design.transform(valid[MODEL_FEATURES])
        target_variants = build_targets(train, y_continuous.loc[train_mask])
        truth_train = y_binary.loc[train_mask].to_numpy()
        truth_valid = y_binary.loc[valid_mask].to_numpy()

        for name in TARGET_NAMES:
            model = Ridge(alpha=100.0, solver="lsqr")
            model.fit(train_matrix, target_variants[name])
            train_score = model.predict(train_matrix)
            valid_score = model.predict(valid_matrix)
            train_prediction = train_score > 0.0
            prediction = valid_score > 0.0
            probability = probability_from_score(valid_score)
            row = {
                "fold": fold,
                "target_variant": name,
                "n_train": int(train_mask.sum()),
                "n_valid": int(valid_mask.sum()),
                "train_accuracy_raw_sign": float((train_prediction == truth_train).mean()),
                "valid_accuracy": float((prediction == truth_valid).mean()),
                "valid_auc": float(roc_auc_score(truth_valid, valid_score)),
                "valid_brier_scaled": float(brier_score_loss(truth_valid, probability)),
                "positive_prediction_rate": float(prediction.mean()),
                "score_std": float(valid_score.std()),
            }
            fold_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            oof_rows[name].append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "TS": valid.TS,
                        "ALLOCATION": valid.ALLOCATION,
                        "GROUP": valid.GROUP,
                        "y_true_binarized": truth_valid,
                        "score": valid_score,
                        "prediction": prediction.astype(np.int8),
                        "is_correct": (prediction == truth_valid).astype(np.int8),
                    },
                    index=valid.index,
                )
            )

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(output_dir / "fold_results.csv", index=False)
    oof_by_name = {name: pd.concat(parts).sort_index() for name, parts in oof_rows.items()}
    for name, oof in oof_by_name.items():
        oof.rename_axis("ROW_ID").to_csv(output_dir / f"oof_{name}.csv")

    baseline = oof_by_name["raw"]
    comparison_rows = []
    for name, oof in oof_by_name.items():
        paired = paired_summary(baseline, oof)
        comparison_rows.append(
            {
                "target_variant": name,
                "accuracy": float(oof.is_correct.mean()),
                "auc": float(roc_auc_score(oof.y_true_binarized, oof.score)),
                "positive_prediction_rate": float(oof.prediction.mean()),
                **paired,
            }
        )
    comparisons = pd.DataFrame(comparison_rows).sort_values("accuracy", ascending=False)
    comparisons.to_csv(output_dir / "comparisons.csv", index=False)
    challenger = comparisons[comparisons.target_variant.ne("raw")].iloc[0]
    baseline_auc = float(comparisons.loc[comparisons.target_variant.eq("raw"), "auc"].iloc[0])
    required_wins = min(max_folds, max(3, int(np.ceil(max_folds * 5 / 8))))
    authorized = bool(
        challenger.gain_accuracy > 0.0002
        and challenger.folds_won >= required_wins
        and challenger.auc >= baseline_auc - 0.0002
    )
    summary: dict[str, object] = {
        "model": "V1 FixedEffectDesign + Ridge(alpha=100)",
        "random_state": random_state,
        "n_folds": max_folds,
        "baseline_accuracy": float(baseline.is_correct.mean()),
        "baseline_auc": baseline_auc,
        "best_challenger": str(challenger.target_variant),
        "best_challenger_accuracy": float(challenger.accuracy),
        "best_challenger_auc": float(challenger.auc),
        "gain_accuracy": float(challenger.gain_accuracy),
        "date_paired_standard_error": float(challenger.date_paired_standard_error),
        "ci95_low": float(challenger.ci95_low),
        "ci95_high": float(challenger.ci95_high),
        "folds_won": int(challenger.folds_won),
        "full_eight_fold_authorized": authorized,
        "target_variants": TARGET_NAMES,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-folds", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.output_dir, args.max_folds, args.random_state), indent=2))

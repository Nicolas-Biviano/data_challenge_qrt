"""H15: very low-dimensional hierarchical allocation responses."""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt import h11_conditional_allocation_response as h11


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h15_low_dimensional_hierarchy"
THREE_STATES = [
    "market_ret1_mean",
    "market_ret1_dispersion",
    "group_ret1_relative",
]
TWO_DIRECTIONAL_STATES = [
    "market_ret1_mean",
    "group_ret1_relative",
]
H15_EXPERIMENTS: dict[str, dict[str, object]] = {
    "baseline_raw": {
        "states": [],
        "group_scale": 0.0,
        "allocation_scale": 0.0,
        "style_scale": 0.0,
    },
    "allocation_lowdim": {
        "states": THREE_STATES,
        "group_scale": 0.0,
        "allocation_scale": 0.20,
        "style_scale": 0.0,
    },
    "hierarchical_ultra": {
        "states": THREE_STATES,
        "group_scale": 0.15,
        "allocation_scale": 0.05,
        "style_scale": 0.0,
    },
    "hierarchical_directional": {
        "states": TWO_DIRECTIONAL_STATES,
        "group_scale": 0.25,
        "allocation_scale": 0.08,
        "style_scale": 0.0,
    },
    "hierarchical_strong": {
        "states": THREE_STATES,
        "group_scale": 0.30,
        "allocation_scale": 0.10,
        "style_scale": 0.0,
    },
}


def run(
    output_dir: Path,
    n_splits: int,
    random_state: int,
    confirm_only: bool = False,
) -> dict[str, object]:
    h11.EXPERIMENTS.update(H15_EXPERIMENTS)
    names = (
        ["baseline_raw", "hierarchical_strong"]
        if confirm_only
        else list(H15_EXPERIMENTS)
    )
    summary = h11.run(names, output_dir, n_splits=n_splits, random_state=random_state)
    results = pd.DataFrame(summary["results"])
    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    challengers = results[results.experiment.ne("baseline_raw")].copy()
    best = challengers.sort_values("gain_accuracy", ascending=False).iloc[0]
    auc_means = fold_metrics.groupby("experiment").auc.mean()
    gaps = fold_metrics.assign(
        gap=fold_metrics.train_accuracy - fold_metrics.accuracy
    ).groupby("experiment").gap.mean()
    baseline_auc = float(auc_means.loc["baseline_raw"])
    required_wins = min(n_splits, max(3, ceil(n_splits * 5 / 8)))
    authorized = bool(
        best.gain_accuracy > 0.0002
        and best.folds_won >= required_wins
        and auc_means.loc[best.experiment] >= baseline_auc - 0.0002
        and gaps.loc[best.experiment] < 0.01
    )
    summary.update(
        {
            "hypothesis": "H15 low-dimensional hierarchical allocation response",
            "random_state": random_state,
            "confirm_only": confirm_only,
            "preregistered_experiments": H15_EXPERIMENTS,
            "best_screening_experiment": str(best.experiment),
            "best_screening_gain": float(best.gain_accuracy),
            "best_screening_auc": float(auc_means.loc[best.experiment]),
            "baseline_auc": baseline_auc,
            "best_screening_gap": float(gaps.loc[best.experiment]),
            "full_eight_fold_authorized": authorized,
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--confirm-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(
        args.output_dir,
        args.n_splits,
        args.random_state,
        confirm_only=args.confirm_only,
    )
    compact = {
        key: result[key]
        for key in (
            "hypothesis",
            "n_splits",
            "best_screening_experiment",
            "best_screening_gain",
            "best_screening_auc",
            "baseline_auc",
            "best_screening_gap",
            "full_eight_fold_authorized",
        )
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))

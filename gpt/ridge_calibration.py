"""Calibrage strict de la branche Ridge, sans target encoding.

Le script fait varier la regularisation globale et la penalisation relative des
pentes ``ALLOCATION x RET_1``. Les folds restent identiques a ceux de la V1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.model import MODEL_FEATURES, FixedEffectDesign, penalized_score  # noqa: E402
from src.cross_validation import CVconfig, ModelConfig, run_cv  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


def make_preprocessor(interaction_scale: float):
    """Retourne un preprocessing avec penalisation relative configurable."""
    def preprocess(X_train: pd.DataFrame, X_valid: pd.DataFrame):
        design = FixedEffectDesign(interaction_scale=interaction_scale).fit(X_train)
        return design.transform(X_train), design.transform(X_valid)

    return preprocess


def run_grid(
    alphas: list[float],
    interaction_scales: list[float],
    output_dir: Path,
) -> pd.DataFrame:
    """Evalue la grille sur les huit folds fixes et exporte les OOF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    rows = []
    fold_rows = []

    for alpha in alphas:
        for interaction_scale in interaction_scales:
            model_config = ModelConfig(
                model=Ridge(alpha=alpha, solver="lsqr"),
                regression=True,
                features=MODEL_FEATURES,
                preprocessing=make_preprocessor(interaction_scale),
            )
            output = run_cv(
                X,
                y,
                model_config,
                CVconfig(n_splits=8, shuffle=True, random_state=0),
            )
            ts_score = penalized_score(output.oof_results, "TS")
            allocation_score = penalized_score(output.oof_results, "ALLOCATION")
            name = f"alpha_{alpha:g}_interaction_{interaction_scale:g}"
            row = {
                "experiment": name,
                "alpha": alpha,
                "interaction_scale": interaction_scale,
                "accuracy": ts_score["accuracy"],
                "ts_standard_error": ts_score["standard_error"],
                "ts_penalized_score": ts_score["score"],
                "allocation_standard_error": allocation_score["standard_error"],
            }
            rows.append(row)
            for fold in output.fold_results:
                fold_rows.append(
                    {
                        "experiment": name,
                        "fold": fold.fold_id,
                        "accuracy": fold.valid_metrics["accuracy"],
                    }
                )
            oof = output.oof_results.copy()
            oof.index.name = "ROW_ID"
            oof.to_csv(output_dir / f"oof_{name}.csv")
            pd.DataFrame(rows).to_csv(output_dir / "calibration_results.csv", index=False)
            pd.DataFrame(fold_rows).to_csv(output_dir / "fold_results.csv", index=False)
            print(row, flush=True)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs="+", type=float, default=[10, 30, 100, 300, 1000])
    parser.add_argument("--interaction-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "ridge_calibration",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_grid(args.alphas, args.interaction_scales, args.output_dir)
    print(result.to_string(index=False))


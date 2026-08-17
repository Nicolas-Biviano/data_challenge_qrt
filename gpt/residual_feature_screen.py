"""Classe les features candidates par correlation aux residus OOF de la V1.

Le classement est separe entre folds 1-4 et folds 5-8 afin d'eviter de retenir
une correlation qui ne se reproduit pas. Il s'agit de selection de variables,
pas de target encoding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS, build_features  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


def finite_correlation(left: pd.Series, right: pd.Series) -> float:
    """Correlation de Pearson sur les valeurs finies communes."""
    x = left.to_numpy(float)
    y = right.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def screen_features(baseline_oof: Path, output_path: Path) -> pd.DataFrame:
    """Mesure la complementarite lineaire de chaque candidate."""
    X_raw = ChallengeDataLoader.load_X_train()
    X = build_features(X_raw)
    oof = pd.read_csv(baseline_oof, index_col="ROW_ID")
    oof = oof.reindex(X.index)
    residual = oof["y_true"] - oof["score"]
    discovery = oof["fold"].le(4)
    confirmation = oof["fold"].ge(5)

    excluded = {"TS", "ALLOCATION", "GROUP", *BASE_RETURNS}
    candidates = [
        column
        for column in X.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(X[column])
    ]
    rows = []
    for column in candidates:
        corr_discovery = finite_correlation(X.loc[discovery, column], residual.loc[discovery])
        corr_confirmation = finite_correlation(
            X.loc[confirmation, column],
            residual.loc[confirmation],
        )
        same_sign = bool(
            np.isfinite(corr_discovery)
            and np.isfinite(corr_confirmation)
            and np.sign(corr_discovery) == np.sign(corr_confirmation)
        )
        stable_strength = (
            min(abs(corr_discovery), abs(corr_confirmation)) if same_sign else 0.0
        )
        rows.append(
            {
                "feature": column,
                "corr_residual_folds_1_4": corr_discovery,
                "corr_residual_folds_5_8": corr_confirmation,
                "same_sign": same_sign,
                "stable_strength": stable_strength,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["stable_strength", "feature"],
        ascending=[False, True],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-oof",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "v1_recheck" / "oof_predictions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "residual_screen" / "feature_ranking.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ranking = screen_features(args.baseline_oof, args.output)
    print(ranking.head(25).to_string(index=False))


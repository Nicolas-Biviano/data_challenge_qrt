"""Tests confirmatoires de features individuelles sur la Ridge V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import (  # noqa: E402
    BASE_RETURNS,
    CATEGORICAL_FEATURES,
    SparseRidgeDesign,
    build_features,
    score_oof,
)
from src.cross_validation import CVconfig, ModelConfig, run_cv  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


CANDIDATES = [
    "group_date_std_SIGNED_VOLUME_1",
    "date_mean_RET_1",
    "group_date_z_RET_1",
    "group_date_rank_RET_1",
    "date_z_RET_1",
    "RET_16",
    "SIGNED_VOLUME_11",
]

TESTS = {
    "baseline": BASE_RETURNS,
    **{f"add_{feature}": BASE_RETURNS + [feature] for feature in CANDIDATES},
    "stable_combo": BASE_RETURNS + CANDIDATES,
    "add_group_date_std_SIGNED_VOLUME_2": BASE_RETURNS
    + ["group_date_std_SIGNED_VOLUME_2"],
    "add_date_count_SIGNED_VOLUME_1": BASE_RETURNS
    + ["date_count_SIGNED_VOLUME_1"],
    "add_group_date_count_SIGNED_VOLUME_1": BASE_RETURNS
    + ["group_date_count_SIGNED_VOLUME_1"],
    "sv1_std_plus_count": BASE_RETURNS
    + ["group_date_std_SIGNED_VOLUME_1", "date_count_SIGNED_VOLUME_1"],
    "sv2_std_plus_count": BASE_RETURNS
    + ["group_date_std_SIGNED_VOLUME_2", "date_count_SIGNED_VOLUME_1"],
}


def make_preprocessor(numeric_features: list[str]):
    """Callback sparse avec transformations apprises dans le fold."""
    def preprocess(X_train: pd.DataFrame, X_valid: pd.DataFrame):
        design = SparseRidgeDesign(numeric_features).fit(X_train)
        return design.transform(X_train), design.transform(X_valid)

    return preprocess


def run_tests(names: list[str], output_dir: Path) -> pd.DataFrame:
    """Execute les tests demandes et exporte scores et predictions OOF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw = ChallengeDataLoader.load_X_train()
    X = build_features(X_raw)
    y = ChallengeDataLoader.load_y_train()
    results = []
    fold_rows = []

    for name in names:
        numeric_features = TESTS[name]
        output = run_cv(
            X,
            y,
            ModelConfig(
                model=Ridge(alpha=100.0, solver="lsqr"),
                regression=True,
                features=CATEGORICAL_FEATURES + numeric_features,
                preprocessing=make_preprocessor(numeric_features),
            ),
            CVconfig(n_splits=8, shuffle=True, random_state=0),
        )
        row = {
            "experiment": name,
            "n_numeric_features": len(numeric_features),
            **score_oof(output.oof_results),
        }
        results.append(row)
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
        pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output_dir / "fold_results.csv", index=False)
        print(row, flush=True)
    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", nargs="+", choices=list(TESTS), default=list(TESTS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "targeted_features",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    table = run_tests(args.tests, args.output_dir)
    print(table.sort_values("accuracy", ascending=False).to_string(index=False))

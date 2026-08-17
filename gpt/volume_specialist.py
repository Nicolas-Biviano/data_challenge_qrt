"""Specialiste de SIGNED_VOLUME_1 avec fallback sur la logistique V2.

Le specialiste est ajuste uniquement sur les lignes ou SIGNED_VOLUME_1 est
observe et n'est utilise que pour ces lignes. Ailleurs, la prediction provient
de la V2 ajustee sur toutes les lignes du fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from gpt.volume_two_part import (  # noqa: E402
    AVAILABILITY_FEATURES,
    CATEGORICAL_FEATURES,
    RANK_FEATURES,
    RETURN_INTERACTIONS,
    VALUE_FEATURES,
    VolumeDesign,
    build_volume_features,
    score_summary,
)
from src.dataloader import ChallengeDataLoader  # noqa: E402


SPECIALISTS = {
    "observed_returns": BASE_RETURNS,
    "observed_rank": BASE_RETURNS + VALUE_FEATURES + RANK_FEATURES,
    "observed_full": (
        BASE_RETURNS
        + AVAILABILITY_FEATURES[1:]
        + VALUE_FEATURES
        + RANK_FEATURES
        + RETURN_INTERACTIONS
    ),
}


def logistic(c: float) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        penalty="l2",
        solver="lbfgs",
        max_iter=200,
        tol=1e-5,
        random_state=0,
    )


def run_specialists(
    names: list[str],
    regularizations: list[float],
    max_folds: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare des experts conditionnels sur les folds de dates fixes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = build_volume_features(ChallengeDataLoader.load_X_train())
    y = ChallengeDataLoader.load_y_train()
    unique_ts = X["TS"].unique()
    splits = KFold(n_splits=8, shuffle=True, random_state=0).split(unique_ts)
    states = {
        (name, c): {
            "score": pd.Series(np.nan, index=X.index),
            "prediction": pd.Series(np.nan, index=X.index),
            "fold": pd.Series(np.nan, index=X.index),
            "fold_rows": [],
        }
        for name in names
        for c in regularizations
    }
    baseline_state = {
        "score": pd.Series(np.nan, index=X.index),
        "prediction": pd.Series(np.nan, index=X.index),
        "fold": pd.Series(np.nan, index=X.index),
    }

    for fold, (train_date_idx, valid_date_idx) in enumerate(splits, start=1):
        if fold > max_folds:
            break
        train_mask = X["TS"].isin(unique_ts[train_date_idx])
        valid_mask = X["TS"].isin(unique_ts[valid_date_idx])
        observed_train = train_mask & X["sv1_available"].eq(1.0)
        observed_valid = valid_mask & X["sv1_available"].eq(1.0)

        base_columns = CATEGORICAL_FEATURES + BASE_RETURNS
        base_design = VolumeDesign(BASE_RETURNS).fit(X.loc[train_mask, base_columns])
        base_train = base_design.transform(X.loc[train_mask, base_columns])
        base_valid = base_design.transform(X.loc[valid_mask, base_columns])
        base_model = logistic(0.003)
        base_model.fit(base_train, y.loc[train_mask, "target_binarized"])
        base_probability = base_model.predict_proba(base_valid)[:, 1]
        baseline_state["score"].loc[valid_mask] = base_probability
        baseline_state["prediction"].loc[valid_mask] = (base_probability > 0.5).astype(
            "int8"
        )
        baseline_state["fold"].loc[valid_mask] = fold

        for name in names:
            numeric_features = SPECIALISTS[name]
            columns = CATEGORICAL_FEATURES + numeric_features
            design = VolumeDesign(numeric_features).fit(
                X.loc[observed_train, columns]
            )
            specialist_train = design.transform(X.loc[observed_train, columns])
            specialist_valid = design.transform(X.loc[observed_valid, columns])
            for c in regularizations:
                model = logistic(c)
                model.fit(
                    specialist_train,
                    y.loc[observed_train, "target_binarized"],
                )
                routed_probability = base_probability.copy()
                local_probability = model.predict_proba(specialist_valid)[:, 1]
                valid_positions = np.flatnonzero(
                    X.loc[valid_mask, "sv1_available"].to_numpy(dtype=bool)
                )
                routed_probability[valid_positions] = local_probability
                routed_prediction = (routed_probability > 0.5).astype("int8")

                state = states[(name, c)]
                state["score"].loc[valid_mask] = routed_probability
                state["prediction"].loc[valid_mask] = routed_prediction
                state["fold"].loc[valid_mask] = fold
                truth = y.loc[valid_mask, "target_binarized"].to_numpy()
                correct = routed_prediction == truth
                observed_positions = X.loc[
                    valid_mask, "sv1_available"
                ].to_numpy(dtype=bool)
                fold_row = {
                    "experiment": name,
                    "C": c,
                    "fold": fold,
                    "accuracy": float(correct.mean()),
                    "accuracy_observed": float(correct[observed_positions].mean()),
                    "accuracy_missing": float(correct[~observed_positions].mean()),
                    "n_observed_valid": int(observed_positions.sum()),
                }
                state["fold_rows"].append(fold_row)
                print(
                    f"{name} C={c:g} fold={fold} "
                    f"all={fold_row['accuracy']:.6f} "
                    f"observed={fold_row['accuracy_observed']:.6f}",
                    flush=True,
                )

    results = []
    fold_rows = []
    baseline_summary = score_summary(baseline_state["prediction"], X, y)
    for (name, c), state in states.items():
        summary = score_summary(state["prediction"], X, y)
        results.append(
            {
                "experiment": name,
                "C": c,
                "n_folds": len(state["fold_rows"]),
                **summary,
                "gain_vs_v2": float(summary["accuracy"] - baseline_summary["accuracy"]),
            }
        )
        fold_rows.extend(state["fold_rows"])
        covered = state["prediction"].notna()
        pd.DataFrame(
            {
                "fold": state["fold"].loc[covered],
                "TS": X.loc[covered, "TS"],
                "ALLOCATION": X.loc[covered, "ALLOCATION"],
                "available": X.loc[covered, "sv1_available"],
                "y_true_binarized": y.loc[covered, "target_binarized"],
                "score": state["score"].loc[covered],
                "prediction": state["prediction"].loc[covered],
            }
        ).rename_axis("ROW_ID").to_csv(
            output_dir / f"oof_{name}_C_{c:g}.csv"
        )

    table = pd.DataFrame(results).sort_values(
        ["accuracy", "ts_penalized_score"], ascending=False
    )
    table.to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_results.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "regularizations": regularizations,
                "specialists": names,
                "max_folds": max_folds,
                "fallback_C": 0.003,
                "target_encoding": False,
                "date_order_used": False,
                "baseline": baseline_summary,
            },
            stream,
            indent=2,
        )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specialists",
        nargs="+",
        choices=list(SPECIALISTS),
        default=list(SPECIALISTS),
    )
    parser.add_argument(
        "--regularizations",
        nargs="+",
        type=float,
        default=[0.003, 0.01, 0.03],
    )
    parser.add_argument("--max-folds", type=int, choices=range(1, 9), default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "volume_specialist_screen",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_specialists(
        args.specialists,
        args.regularizations,
        args.max_folds,
        args.output_dir,
    )
    print(result.to_string(index=False))

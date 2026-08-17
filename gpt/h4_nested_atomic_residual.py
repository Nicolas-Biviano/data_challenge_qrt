"""H4-G: sélection atomique stable puis forward conditionnelle imbriquée."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from gpt.h4_nested_forward_selection import (
    CachedForwardDesign,
    candidate_groups,
    fit_predict,
    prepare_inner_caches,
)
from gpt.h4_nonlinear_group_features import REFERENCE_PATH, build_h4_features
from gpt.hypotheses_h1_h2_h3 import cluster_bootstrap_models
from src.dataloader import ChallengeDataLoader


def atomic_design_groups(families: dict[str, list[str]]) -> dict[str, list[str]]:
    """Une colonne par candidat ; les familles ne sont jamais concaténées."""
    return {feature: [feature] for features in families.values() for feature in features}


def redundancy_families() -> dict[str, list[str]]:
    """Huit familles : les formes alternatives d'une coordonnée sont concurrentes."""
    old_groups = candidate_groups()
    return {
        "ret1_date_z": [
            old_groups[name][0]
            for name in [
                "ret1_date_z_linear",
                "ret1_date_z_tanh05",
                "ret1_date_z_arctan",
                "ret1_date_z_signed_log",
            ]
        ],
        "ret1_group_z": [
            old_groups[name][0]
            for name in [
                "ret1_group_z_linear",
                "ret1_group_z_tanh05",
                "ret1_group_z_arctan",
                "ret1_group_z_signed_log",
            ]
        ],
        "ret1_date_rank": old_groups["ret1_date_rank_shape"],
        "ret1_group_rank": old_groups["ret1_group_rank_shape"],
        "turnover_date_rank": old_groups["turnover_date_rank_shape"],
        "turnover_group_rank": old_groups["turnover_group_rank_shape"],
        "sv1_availability": old_groups["sv1_available"],
        "sv_positive_share": old_groups["sv_positive_share_shape"],
    }


def evaluate_set(
    caches: list[dict[str, object]],
    selected: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    truth_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    complete_parts: list[np.ndarray] = []
    rows = []
    for cache in caches:
        score, n_features, n_iter = fit_predict(
            cache["train"], cache["valid"], selected, cache["y_train"]
        )
        truth = cache["y_valid"]
        complete = cache["valid_complete"]
        prediction = score > 0.5
        truth_parts.append(truth)
        score_parts.append(score)
        complete_parts.append(complete)
        rows.append(
            {
                "inner_fold": int(cache["fold"]),
                "accuracy": float(accuracy_score(truth, prediction)),
                "accuracy_complete": float(
                    accuracy_score(truth[complete], prediction[complete])
                ),
                "auc": float(roc_auc_score(truth, score)),
                "positive_rate": float(prediction.mean()),
                "n_features": n_features,
                "n_iter": n_iter,
            }
        )
    truth = np.concatenate(truth_parts)
    score = np.concatenate(score_parts)
    complete = np.concatenate(complete_parts)
    prediction = score > 0.5
    pooled = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "accuracy_complete": float(accuracy_score(truth[complete], prediction[complete])),
        "auc": float(roc_auc_score(truth, score)),
        "positive_rate": float(prediction.mean()),
    }
    return pooled, pd.DataFrame(rows)


def comparison_row(
    candidate_metrics: dict[str, float],
    candidate_folds: pd.DataFrame,
    current_metrics: dict[str, float],
    current_folds: pd.DataFrame,
) -> dict[str, float | int]:
    fold_gain = candidate_folds.set_index("inner_fold")[["accuracy", "accuracy_complete"]].sub(
        current_folds.set_index("inner_fold")[["accuracy", "accuracy_complete"]]
    )
    global_gains = fold_gain["accuracy"]
    complete_gains = fold_gain["accuracy_complete"]
    n_folds = len(fold_gain)
    se_global = float(global_gains.std(ddof=1) / np.sqrt(n_folds)) if n_folds > 1 else 0.0
    se_complete = float(complete_gains.std(ddof=1) / np.sqrt(n_folds)) if n_folds > 1 else 0.0
    mean_global = float(global_gains.mean())
    mean_complete = float(complete_gains.mean())
    return {
        "accuracy": candidate_metrics["accuracy"],
        "accuracy_complete": candidate_metrics["accuracy_complete"],
        "auc": candidate_metrics["auc"],
        "positive_rate": candidate_metrics["positive_rate"],
        "gain_global": candidate_metrics["accuracy"] - current_metrics["accuracy"],
        "gain_complete": candidate_metrics["accuracy_complete"]
        - current_metrics["accuracy_complete"],
        "mean_fold_gain_global": mean_global,
        "se_fold_gain_global": se_global,
        "mean_fold_gain_complete": mean_complete,
        "se_fold_gain_complete": se_complete,
        "stable_global": mean_global - se_global,
        "stable_complete": mean_complete - se_complete,
        "stable_objective": min(mean_global - se_global, mean_complete - se_complete),
        "positive_folds_global": int(global_gains.gt(0).sum()),
        "positive_folds_complete": int(complete_gains.gt(0).sum()),
    }


def choose_representatives(
    outer_fold: int,
    caches: list[dict[str, object]],
    families: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, tuple[dict[str, float], pd.DataFrame]], list[dict[str, object]]]:
    baseline_metrics, baseline_folds = evaluate_set(caches, [])
    evaluations: dict[str, tuple[dict[str, float], pd.DataFrame]] = {}
    trace: list[dict[str, object]] = []
    representatives: dict[str, str] = {}
    for family, features in families.items():
        family_rows = []
        for feature in features:
            metrics, folds = evaluate_set(caches, [feature])
            evaluations[feature] = (metrics, folds)
            row = {
                "outer_fold": outer_fold,
                "family": family,
                "feature": feature,
                **comparison_row(metrics, folds, baseline_metrics, baseline_folds),
            }
            family_rows.append(row)
        winner = max(
            family_rows,
            key=lambda row: (
                row["stable_objective"],
                min(row["gain_global"], row["gain_complete"]),
            ),
        )
        representatives[family] = str(winner["feature"])
        for row in family_rows:
            row["is_representative"] = row["feature"] == winner["feature"]
        trace.extend(family_rows)
        print(
            f"outer={outer_fold} representative family={family} feature={winner['feature']} "
            f"stable={100*winner['stable_objective']:+.4f}pp",
            flush=True,
        )
    evaluations["__baseline__"] = (baseline_metrics, baseline_folds)
    return representatives, evaluations, trace


def select_residual_features(
    outer_fold: int,
    caches: list[dict[str, object]],
    representatives: dict[str, str],
    baseline_evaluations: dict[str, tuple[dict[str, float], pd.DataFrame]],
    max_steps: int,
    min_positive_folds: int,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    selected_features: list[str] = []
    selected_families: list[str] = []
    trace: list[dict[str, object]] = []
    current_metrics, current_folds = baseline_evaluations["__baseline__"]
    for step in range(1, max_steps + 1):
        rows = []
        cache_by_family: dict[str, tuple[dict[str, float], pd.DataFrame]] = {}
        for family, feature in representatives.items():
            if family in selected_families:
                continue
            if step == 1:
                metrics, folds = baseline_evaluations[feature]
            else:
                metrics, folds = evaluate_set(caches, selected_features + [feature])
            cache_by_family[family] = (metrics, folds)
            row = {
                "outer_fold": outer_fold,
                "step": step,
                "family": family,
                "feature": feature,
                "selected_before": "|".join(selected_features),
                **comparison_row(metrics, folds, current_metrics, current_folds),
            }
            rows.append(row)
        winner = max(
            rows,
            key=lambda row: (
                row["stable_objective"],
                min(row["gain_global"], row["gain_complete"]),
            ),
        )
        accepted = (
            winner["gain_global"] > 0
            and winner["gain_complete"] > 0
            and winner["positive_folds_global"] >= min_positive_folds
            and winner["positive_folds_complete"] >= min_positive_folds
        )
        for row in rows:
            row["winner"] = row["family"] == winner["family"]
            row["accepted"] = bool(accepted and row["winner"])
        trace.extend(rows)
        print(
            f"outer={outer_fold} step={step} winner={winner['feature']} "
            f"gain={100*winner['gain_global']:+.4f}pp "
            f"complete={100*winner['gain_complete']:+.4f}pp "
            f"positive_folds={winner['positive_folds_global']}/"
            f"{winner['positive_folds_complete']} accepted={accepted}",
            flush=True,
        )
        if not accepted:
            break
        selected_features.append(str(winner["feature"]))
        selected_families.append(str(winner["family"]))
        current_metrics, current_folds = cache_by_family[str(winner["family"])]
    return selected_features, selected_families, trace


def run(
    output_dir: Path,
    outer_splits: int = 8,
    inner_splits: int = 3,
    max_steps: int = 3,
    min_positive_folds: int = 2,
    max_outer_folds: int = 8,
) -> dict[str, object]:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    X, _ = build_h4_features(X_raw)
    X["date_n_rows"] = X.TS.map(X.groupby("TS", observed=True).size()).astype("int16")
    families = redundancy_families()
    atomic_groups = atomic_design_groups(families)

    unique_dates = X.TS.unique()
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=0)
    oof = pd.DataFrame(
        {
            "fold": np.nan,
            "TS": X.TS,
            "ALLOCATION": X.ALLOCATION,
            "y_true": y.target,
            "y_true_binarized": y.target_binarized,
            "score": np.nan,
            "prediction": np.nan,
        },
        index=X.index,
    )
    representative_trace: list[dict[str, object]] = []
    forward_trace: list[dict[str, object]] = []
    outer_rows = []
    selected_features_by_fold: dict[int, list[str]] = {}
    selected_families_by_fold: dict[int, list[str]] = {}

    for outer_fold, (train_date_idx, valid_date_idx) in enumerate(outer.split(unique_dates), start=1):
        if outer_fold > max_outer_folds:
            break
        train_dates = unique_dates[train_date_idx]
        valid_dates = unique_dates[valid_date_idx]
        print(f"Preparing outer fold {outer_fold}/{max_outer_folds}", flush=True)
        caches = prepare_inner_caches(X, y, train_dates, atomic_groups, inner_splits)
        representatives, baseline_evaluations, rep_rows = choose_representatives(
            outer_fold, caches, families
        )
        selected_features, selected_families, selection_rows = select_residual_features(
            outer_fold,
            caches,
            representatives,
            baseline_evaluations,
            max_steps,
            min_positive_folds,
        )
        representative_trace.extend(rep_rows)
        forward_trace.extend(selection_rows)
        selected_features_by_fold[outer_fold] = selected_features
        selected_families_by_fold[outer_fold] = selected_families
        del caches, baseline_evaluations
        gc.collect()

        train_mask = X.TS.isin(train_dates)
        valid_mask = X.TS.isin(valid_dates)
        design = CachedForwardDesign(atomic_groups).fit(X.loc[train_mask])
        train_parts = design.transform(X.loc[train_mask])
        valid_parts = design.transform(X.loc[valid_mask])
        score, n_features, n_iter = fit_predict(
            train_parts,
            valid_parts,
            selected_features,
            y.loc[train_mask, "target_binarized"].astype(int),
        )
        truth = y.loc[valid_mask, "target_binarized"].to_numpy(int)
        prediction = (score > 0.5).astype("int8")
        oof.loc[valid_mask, "fold"] = outer_fold
        oof.loc[valid_mask, "score"] = score
        oof.loc[valid_mask, "prediction"] = prediction
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_families": "|".join(selected_families),
                "selected_features": "|".join(selected_features),
                "n_selected_features": len(selected_features),
                "n_features": n_features,
                "n_iter": n_iter,
                "accuracy": accuracy_score(truth, prediction),
                "auc": roc_auc_score(truth, score),
                "positive_rate": prediction.mean(),
            }
        )
        print(
            f"OUTER fold={outer_fold} selected={selected_features} "
            f"accuracy={outer_rows[-1]['accuracy']:.6f}",
            flush=True,
        )
        del design, train_parts, valid_parts
        gc.collect()

    covered = oof.prediction.notna()
    oof.loc[covered, "is_correct"] = oof.loc[covered, "prediction"].eq(
        oof.loc[covered, "y_true_binarized"]
    ).astype(float)
    oof.index.name = "ROW_ID"
    oof.to_csv(output_dir / "oof_nested_atomic_residual.csv")
    pd.DataFrame(representative_trace).to_csv(
        output_dir / "representative_trace.csv", index=False
    )
    pd.DataFrame(forward_trace).to_csv(output_dir / "forward_trace.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(output_dir / "outer_fold_results.csv", index=False)

    reference = pd.read_csv(REFERENCE_PATH, index_col="ROW_ID").loc[covered]
    candidate = oof.loc[covered]
    covered_dates = pd.Index(candidate.TS.unique())
    date_sizes = X.groupby("TS", observed=True).size()
    complete_dates = date_sizes[date_sizes.eq(276)].index.intersection(covered_dates)
    uncertainty_global = cluster_bootstrap_models(
        {"V2": reference, "H4-G atomic residual": candidate},
        covered_dates,
        reference="V2",
    )
    uncertainty_global["scope"] = "Toutes dates"
    uncertainty_complete = cluster_bootstrap_models(
        {"V2": reference, "H4-G atomic residual": candidate},
        complete_dates,
        reference="V2",
    )
    uncertainty_complete["scope"] = "Dates complètes (276)"
    uncertainty = pd.concat([uncertainty_global, uncertainty_complete], ignore_index=True)
    uncertainty.to_csv(output_dir / "uncertainty.csv", index=False)

    truth = candidate.y_true_binarized.astype(int)
    prediction = candidate.prediction.astype(int)
    metrics = {
        "n_rows": int(len(candidate)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "auc": float(roc_auc_score(truth, candidate.score)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "accuracy_y0": float(accuracy_score(truth[truth.eq(0)], prediction[truth.eq(0)])),
        "accuracy_y1": float(accuracy_score(truth[truth.eq(1)], prediction[truth.eq(1)])),
        "positive_rate": float(prediction.mean()),
        "accuracy_complete": float(
            candidate[candidate.TS.isin(complete_dates)].is_correct.mean()
        ),
    }
    feature_frequency = (
        pd.Series(
            [feature for selected in selected_features_by_fold.values() for feature in selected],
            dtype="object",
        )
        .value_counts()
        .rename_axis("feature")
        .rename("n_outer_folds_selected")
        .reset_index()
    )
    family_frequency = (
        pd.Series(
            [family for selected in selected_families_by_fold.values() for family in selected],
            dtype="object",
        )
        .value_counts()
        .rename_axis("family")
        .rename("n_outer_folds_selected")
        .reset_index()
    )
    feature_frequency.to_csv(output_dir / "feature_selection_frequency.csv", index=False)
    family_frequency.to_csv(output_dir / "family_selection_frequency.csv", index=False)
    summary = {
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "max_steps": max_steps,
        "min_positive_folds": min_positive_folds,
        "max_outer_folds": max_outer_folds,
        "n_atomic_candidates": len(atomic_groups),
        "families": families,
        "selected_features_by_fold": selected_features_by_fold,
        "selected_families_by_fold": selected_families_by_fold,
        "metrics": metrics,
        "feature_selection_frequency": feature_frequency.to_dict("records"),
        "family_selection_frequency": family_frequency.to_dict("records"),
        "runtime_seconds": float(time.perf_counter() - start),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "metrics": metrics,
                "feature_selection_frequency": summary["feature_selection_frequency"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=8)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--min-positive-folds", type=int, default=2)
    parser.add_argument("--max-outer-folds", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "h4_nested_atomic_residual",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_dir,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        max_steps=args.max_steps,
        min_positive_folds=args.min_positive_folds,
        max_outer_folds=args.max_outer_folds,
    )

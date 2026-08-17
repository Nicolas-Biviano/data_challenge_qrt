"""H4-F: forward selection imbriquée, groupée par dates, des features H4."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from gpt.h4_nonlinear_group_features import (  # noqa: E402
    MODEL_C,
    REFERENCE_PATH,
    build_h4_features,
)
from gpt.hypotheses_h1_h2_h3 import cluster_bootstrap_models  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


CATEGORICAL = ["ALLOCATION", "GROUP"]


def candidate_groups() -> dict[str, list[str]]:
    """Quatorze groupes disjoints, figés avant la sélection."""
    groups = {
        "ret1_date_z_linear": ["ret1_date_z__linear"],
        "ret1_date_z_tanh05": ["ret1_date_z__tanh05"],
        "ret1_date_z_arctan": ["ret1_date_z__arctan"],
        "ret1_date_z_signed_log": ["ret1_date_z__signed_log"],
        "ret1_group_z_linear": ["ret1_group_z__linear"],
        "ret1_group_z_tanh05": ["ret1_group_z__tanh05"],
        "ret1_group_z_arctan": ["ret1_group_z__arctan"],
        "ret1_group_z_signed_log": ["ret1_group_z__signed_log"],
        "ret1_date_rank_shape": [
            "ret1_date_rank__tanh05",
            "ret1_date_rank__tanh1",
            "ret1_date_rank__arctan",
            "ret1_date_rank__signed_log",
            "ret1_date_rank__hinge_20",
            "ret1_date_rank__hinge_50",
            "ret1_date_rank__hinge_80",
        ],
        "ret1_group_rank_shape": [
            "ret1_group_rank__tanh05",
            "ret1_group_rank__tanh1",
            "ret1_group_rank__arctan",
            "ret1_group_rank__signed_log",
            "ret1_group_rank__hinge_20",
            "ret1_group_rank__hinge_50",
            "ret1_group_rank__hinge_80",
        ],
        "turnover_date_rank_shape": [
            "turnover_date_rank__tanh05",
            "turnover_date_rank__tanh1",
            "turnover_date_rank__arctan",
            "turnover_date_rank__signed_log",
            "turnover_date_rank__hinge_20",
            "turnover_date_rank__hinge_50",
            "turnover_date_rank__hinge_80",
        ],
        "turnover_group_rank_shape": [
            "turnover_group_rank__tanh05",
            "turnover_group_rank__tanh1",
            "turnover_group_rank__arctan",
            "turnover_group_rank__signed_log",
            "turnover_group_rank__hinge_20",
            "turnover_group_rank__hinge_50",
            "turnover_group_rank__hinge_80",
        ],
        "sv1_available": ["sv1_available"],
        "sv_positive_share_shape": [
            "sv_positive_share_centered__linear",
            "sv_positive_share_centered__tanh05",
            "sv_positive_share_centered__tanh1",
            "sv_positive_share_centered__arctan",
            "sv_positive_share_centered__signed_log",
        ],
    }
    all_columns = [column for columns in groups.values() for column in columns]
    if len(all_columns) != len(set(all_columns)):
        raise ValueError("Les groupes candidats H4-F doivent être disjoints")
    return groups


@dataclass
class MatrixParts:
    base: sparse.csr_matrix
    groups: dict[str, sparse.csr_matrix]

    def combine(self, selected: list[str]) -> sparse.csr_matrix:
        return sparse.hstack(
            [self.base] + [self.groups[name] for name in selected],
            format="csr",
            dtype=np.float32,
        )


@dataclass
class CachedForwardDesign:
    groups: dict[str, list[str]]

    def __post_init__(self) -> None:
        self.candidate_columns = [column for columns in self.groups.values() for column in columns]
        self.all_numeric = BASE_RETURNS + self.candidate_columns
        self.numeric_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
        self.category_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        self.allocation_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)

    def fit(self, X: pd.DataFrame) -> "CachedForwardDesign":
        self.numeric_pipeline.fit(X[self.all_numeric])
        self.category_encoder.fit(X[CATEGORICAL])
        self.allocation_encoder.fit(X[["ALLOCATION"]])
        return self

    def transform(self, X: pd.DataFrame) -> MatrixParts:
        numeric = self.numeric_pipeline.transform(X[self.all_numeric]).astype(np.float32)
        categories = self.category_encoder.transform(X[CATEGORICAL]).tocsr()
        allocation = self.allocation_encoder.transform(X[["ALLOCATION"]]).tocsr()
        base_numeric = sparse.csr_matrix(numeric[:, : len(BASE_RETURNS)])
        allocation_ret1 = allocation.multiply(numeric[:, 0, None]).tocsr()
        base = sparse.hstack([base_numeric, categories, allocation_ret1], format="csr", dtype=np.float32)
        position = {name: index for index, name in enumerate(self.all_numeric)}
        group_parts = {
            name: sparse.csr_matrix(numeric[:, [position[column] for column in columns]])
            for name, columns in self.groups.items()
        }
        return MatrixParts(base=base, groups=group_parts)


def fit_predict(
    train: MatrixParts,
    valid: MatrixParts,
    selected: list[str],
    y_train: pd.Series,
) -> tuple[np.ndarray, int, int]:
    train_matrix = train.combine(selected)
    valid_matrix = valid.combine(selected)
    model = LogisticRegression(
        C=MODEL_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=180,
        tol=1e-5,
        random_state=0,
    )
    model.fit(train_matrix, y_train)
    probability = model.predict_proba(valid_matrix)[:, 1]
    n_features = train_matrix.shape[1]
    n_iter = int(model.n_iter_.max())
    del train_matrix, valid_matrix, model
    return probability, n_features, n_iter


def prepare_inner_caches(
    X: pd.DataFrame,
    y: pd.DataFrame,
    outer_train_dates: np.ndarray,
    groups: dict[str, list[str]],
    inner_splits: int,
) -> list[dict[str, object]]:
    caches = []
    splitter = KFold(n_splits=inner_splits, shuffle=True, random_state=731)
    for inner_fold, (train_idx, valid_idx) in enumerate(splitter.split(outer_train_dates), start=1):
        train_dates = outer_train_dates[train_idx]
        valid_dates = outer_train_dates[valid_idx]
        train_mask = X.TS.isin(train_dates)
        valid_mask = X.TS.isin(valid_dates)
        design = CachedForwardDesign(groups).fit(X.loc[train_mask])
        caches.append(
            {
                "fold": inner_fold,
                "train": design.transform(X.loc[train_mask]),
                "valid": design.transform(X.loc[valid_mask]),
                "y_train": y.loc[train_mask, "target_binarized"].astype(int),
                "y_valid": y.loc[valid_mask, "target_binarized"].astype(int).to_numpy(),
                "valid_index": X.index[valid_mask],
                "valid_complete": X.loc[valid_mask, "date_n_rows"].eq(276).to_numpy(),
            }
        )
        del design
        gc.collect()
    return caches


def evaluate_set(
    caches: list[dict[str, object]],
    selected: list[str],
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    truth_parts = []
    score_parts = []
    complete_parts = []
    fold_rows = []
    for cache in caches:
        score, n_features, n_iter = fit_predict(
            cache["train"],
            cache["valid"],
            selected,
            cache["y_train"],
        )
        truth = cache["y_valid"]
        prediction = score > .5
        truth_parts.append(truth)
        score_parts.append(score)
        complete_parts.append(cache["valid_complete"])
        fold_rows.append(
            {
                "inner_fold": int(cache["fold"]),
                "accuracy": float(accuracy_score(truth, prediction)),
                "auc": float(roc_auc_score(truth, score)),
                "n_features": n_features,
                "n_iter": n_iter,
            }
        )
    truth = np.concatenate(truth_parts)
    score = np.concatenate(score_parts)
    complete = np.concatenate(complete_parts)
    prediction = score > .5
    metrics = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "accuracy_complete": float(accuracy_score(truth[complete], prediction[complete])),
        "auc": float(roc_auc_score(truth, score)),
        "positive_rate": float(prediction.mean()),
    }
    return metrics, fold_rows


def select_for_outer_fold(
    outer_fold: int,
    caches: list[dict[str, object]],
    groups: dict[str, list[str]],
    max_steps: int,
) -> tuple[list[str], list[dict[str, object]]]:
    selected: list[str] = []
    trace = []
    current, _ = evaluate_set(caches, selected)
    for step in range(1, max_steps + 1):
        candidate_rows = []
        for candidate in groups:
            if candidate in selected:
                continue
            metrics, _ = evaluate_set(caches, selected + [candidate])
            gain_global = metrics["accuracy"] - current["accuracy"]
            gain_complete = metrics["accuracy_complete"] - current["accuracy_complete"]
            objective = min(gain_global, gain_complete)
            row = {
                "outer_fold": outer_fold,
                "step": step,
                "candidate": candidate,
                "selected_before": "|".join(selected),
                **metrics,
                "gain_global": gain_global,
                "gain_complete": gain_complete,
                "objective": objective,
            }
            candidate_rows.append(row)
            print(
                f"outer={outer_fold} step={step} candidate={candidate} "
                f"gain={100*gain_global:+.4f}pp complete={100*gain_complete:+.4f}pp",
                flush=True,
            )
        ranked = sorted(
            candidate_rows,
            key=lambda row: (row["objective"], row["gain_global"] + row["gain_complete"]),
            reverse=True,
        )
        winner = ranked[0]
        for row in candidate_rows:
            row["winner"] = row["candidate"] == winner["candidate"]
            row["accepted"] = False
        accepted = winner["gain_global"] > 0 and winner["gain_complete"] > 0
        if accepted:
            selected.append(winner["candidate"])
            current = {key: winner[key] for key in ["accuracy", "accuracy_complete", "auc", "positive_rate"]}
            next(row for row in candidate_rows if row["candidate"] == winner["candidate"])["accepted"] = True
        trace.extend(candidate_rows)
        print(
            f"outer={outer_fold} step={step} winner={winner['candidate']} accepted={accepted} "
            f"selected={selected}",
            flush=True,
        )
        if not accepted:
            break
    return selected, trace


def run(
    output_dir: Path,
    outer_splits: int = 8,
    inner_splits: int = 3,
    max_steps: int = 3,
    max_outer_folds: int = 8,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    X, _ = build_h4_features(X_raw)
    X["date_n_rows"] = X.TS.map(X.groupby("TS", observed=True).size()).astype("int16")
    groups = candidate_groups()
    missing = [column for columns in groups.values() for column in columns if column not in X]
    if missing:
        raise ValueError(f"Features H4-F absentes: {missing}")

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
    all_trace = []
    outer_rows = []
    selected_by_fold: dict[int, list[str]] = {}
    for outer_fold, (train_date_idx, valid_date_idx) in enumerate(outer.split(unique_dates), start=1):
        if outer_fold > max_outer_folds:
            break
        train_dates = unique_dates[train_date_idx]
        valid_dates = unique_dates[valid_date_idx]
        print(f"Preparing outer fold {outer_fold}/{max_outer_folds}", flush=True)
        caches = prepare_inner_caches(X, y, train_dates, groups, inner_splits)
        selected, trace = select_for_outer_fold(outer_fold, caches, groups, max_steps)
        selected_by_fold[outer_fold] = selected
        all_trace.extend(trace)
        del caches
        gc.collect()

        train_mask = X.TS.isin(train_dates)
        valid_mask = X.TS.isin(valid_dates)
        design = CachedForwardDesign(groups).fit(X.loc[train_mask])
        train_parts = design.transform(X.loc[train_mask])
        valid_parts = design.transform(X.loc[valid_mask])
        score, n_features, n_iter = fit_predict(
            train_parts,
            valid_parts,
            selected,
            y.loc[train_mask, "target_binarized"].astype(int),
        )
        truth = y.loc[valid_mask, "target_binarized"].to_numpy(int)
        prediction = (score > .5).astype("int8")
        oof.loc[valid_mask, "fold"] = outer_fold
        oof.loc[valid_mask, "score"] = score
        oof.loc[valid_mask, "prediction"] = prediction
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected": "|".join(selected),
                "n_selected_groups": len(selected),
                "n_features": n_features,
                "n_iter": n_iter,
                "accuracy": accuracy_score(truth, prediction),
                "auc": roc_auc_score(truth, score),
                "positive_rate": prediction.mean(),
            }
        )
        print(
            f"OUTER fold={outer_fold} selected={selected} accuracy={outer_rows[-1]['accuracy']:.6f}",
            flush=True,
        )
        del design, train_parts, valid_parts
        gc.collect()

    covered = oof.prediction.notna()
    oof.loc[covered, "is_correct"] = oof.loc[covered, "prediction"].eq(
        oof.loc[covered, "y_true_binarized"]
    ).astype(float)
    oof.index.name = "ROW_ID"
    oof.to_csv(output_dir / "oof_nested_forward.csv")
    trace_table = pd.DataFrame(all_trace)
    trace_table.to_csv(output_dir / "selection_trace.csv", index=False)
    outer_table = pd.DataFrame(outer_rows)
    outer_table.to_csv(output_dir / "outer_fold_results.csv", index=False)

    reference = pd.read_csv(REFERENCE_PATH, index_col="ROW_ID").loc[covered]
    candidate = oof.loc[covered]
    covered_dates = pd.Index(candidate.TS.unique())
    date_sizes = X.groupby("TS", observed=True).size()
    complete_dates = date_sizes[date_sizes.eq(276)].index.intersection(covered_dates)
    uncertainty_global = cluster_bootstrap_models(
        {"V2": reference, "H4-F nested forward": candidate},
        covered_dates,
        reference="V2",
    )
    uncertainty_global["scope"] = "Toutes dates"
    uncertainty_complete = cluster_bootstrap_models(
        {"V2": reference, "H4-F nested forward": candidate},
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
        "accuracy_complete": float(candidate[candidate.TS.isin(complete_dates)].is_correct.mean()),
    }
    selection_frequency = (
        pd.Series([name for selected in selected_by_fold.values() for name in selected])
        .value_counts()
        .rename_axis("group")
        .rename("n_outer_folds_selected")
        .reset_index()
    )
    selection_frequency.to_csv(output_dir / "selection_frequency.csv", index=False)
    summary = {
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "max_steps": max_steps,
        "max_outer_folds": max_outer_folds,
        "candidate_groups": groups,
        "selected_by_fold": selected_by_fold,
        "metrics": metrics,
        "selection_frequency": selection_frequency.to_dict("records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"metrics": metrics, "selection_frequency": summary["selection_frequency"]}, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=8)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-outer-folds", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "h4_nested_forward",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_dir,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        max_steps=args.max_steps,
        max_outer_folds=args.max_outer_folds,
    )

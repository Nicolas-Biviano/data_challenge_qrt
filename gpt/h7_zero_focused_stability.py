"""H7: nested stability selection of zero-focused return transformations."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from gpt.h4_nonlinear_group_features import robust_group_z  # noqa: E402
from gpt.hypotheses_h1_h2_h3 import cluster_bootstrap_models  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h7_zero_focused_stability"
REFERENCE_PATH = GPT_DIR / "outputs" / "classification_full" / "oof_logistic_C_0.003.csv"
MODEL_C = 0.003
SELECTION_C = 0.0015
SELECTION_L1_RATIO = 0.8


def build_zero_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    output = X[["TS", "ALLOCATION", "GROUP"] + BASE_RETURNS].copy()
    family: dict[str, str] = {}
    additions: dict[str, pd.Series] = {}
    for feature in BASE_RETURNS:
        z = robust_group_z(X[feature], [X.TS]).astype("float32")
        forms = {
            "signed_sqrt": np.sign(z) * np.sqrt(np.abs(z)),
            "tanh025": np.tanh(z / 0.25),
            "tanh050": np.tanh(z / 0.50),
            "softsign025": z / (0.25 + np.abs(z)),
            "clip025": np.clip(z / 0.25, -1, 1),
        }
        for suffix, values in forms.items():
            name = f"{feature}__zero_{suffix}"
            additions[name] = pd.Series(values, index=X.index, dtype="float32")
            family[name] = feature
    output = pd.concat([output, pd.DataFrame(additions, index=X.index)], axis=1)
    return output, family


@dataclass
class MatrixParts:
    base: sparse.csr_matrix
    candidates: dict[str, sparse.csr_matrix]

    def combine(self, selected: list[str]) -> sparse.csr_matrix:
        return sparse.hstack(
            [self.base] + [self.candidates[name] for name in selected],
            format="csr",
            dtype=np.float32,
        )


class ZeroFocusedDesign:
    def __init__(self, candidates: list[str]):
        self.candidates = candidates
        self.numeric_columns = BASE_RETURNS + candidates
        self.numeric_pipeline = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler()
        )
        self.category_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        self.allocation_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)

    def fit(self, X: pd.DataFrame) -> "ZeroFocusedDesign":
        self.numeric_pipeline.fit(X[self.numeric_columns])
        self.category_encoder.fit(X[["ALLOCATION", "GROUP"]])
        self.allocation_encoder.fit(X[["ALLOCATION"]])
        return self

    def transform(self, X: pd.DataFrame) -> MatrixParts:
        numeric = self.numeric_pipeline.transform(X[self.numeric_columns]).astype(np.float32)
        categories = self.category_encoder.transform(X[["ALLOCATION", "GROUP"]]).tocsr()
        allocation = self.allocation_encoder.transform(X[["ALLOCATION"]]).tocsr()
        base_numeric = sparse.csr_matrix(numeric[:, : len(BASE_RETURNS)])
        allocation_ret1 = allocation.multiply(numeric[:, 0, None]).tocsr()
        base = sparse.hstack(
            [base_numeric, categories, allocation_ret1], format="csr", dtype=np.float32
        )
        positions = {name: len(BASE_RETURNS) + i for i, name in enumerate(self.candidates)}
        candidate_parts = {
            name: sparse.csr_matrix(numeric[:, [position]])
            for name, position in positions.items()
        }
        return MatrixParts(base=base, candidates=candidate_parts)


def stability_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parts: MatrixParts,
    candidates: list[str],
    families: dict[str, str],
    outer_fold: int,
    n_repeats: int,
) -> tuple[list[str], list[dict[str, object]]]:
    rng = np.random.default_rng(20260807 + outer_fold)
    dates = X_train.TS.unique()
    full_matrix = parts.combine(candidates)
    candidate_start = parts.base.shape[1]
    rows: list[dict[str, object]] = []
    for repeat in range(1, n_repeats + 1):
        sampled_dates = rng.choice(dates, size=int(np.ceil(0.60 * len(dates))), replace=False)
        sampled = X_train.TS.isin(sampled_dates).to_numpy()
        model = LogisticRegression(
            C=SELECTION_C,
            penalty="elasticnet",
            l1_ratio=SELECTION_L1_RATIO,
            solver="saga",
            max_iter=80,
            tol=1e-3,
            random_state=1000 * outer_fold + repeat,
            n_jobs=-1,
        )
        model.fit(full_matrix[sampled], y_train.to_numpy()[sampled])
        coefficients = model.coef_[0, candidate_start : candidate_start + len(candidates)]
        nonzero = np.flatnonzero(np.abs(coefficients) > 1e-8)
        ranked = nonzero[np.argsort(np.abs(coefficients[nonzero]))[::-1]][:8]
        selected_positions = set(ranked.tolist())
        for position, candidate in enumerate(candidates):
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "repeat": repeat,
                    "candidate": candidate,
                    "family": families[candidate],
                    "coefficient": float(coefficients[position]),
                    "nonzero": bool(position in nonzero),
                    "selected_top8": bool(position in selected_positions),
                    "n_iter": int(model.n_iter_.max()),
                }
            )
        print(
            f"H7 outer={outer_fold} repeat={repeat}/{n_repeats} "
            f"nonzero={len(nonzero)} n_iter={model.n_iter_.max()}",
            flush=True,
        )
        del model
    trace = pd.DataFrame(rows)
    selected_rows = trace[trace.selected_top8]
    summary = (
        trace.groupby(["candidate", "family"], observed=True)
        .agg(
            selection_frequency=("selected_top8", "mean"),
            nonzero_frequency=("nonzero", "mean"),
            mean_abs_coefficient=("coefficient", lambda x: np.mean(np.abs(x))),
        )
        .reset_index()
    )
    sign = (
        selected_rows.assign(sign=lambda d: np.sign(d.coefficient))
        .groupby("candidate", observed=True)
        .sign.agg(lambda x: max((x > 0).mean(), (x < 0).mean()))
        .rename("sign_stability")
    )
    summary = summary.merge(sign, on="candidate", how="left").fillna({"sign_stability": 0})
    eligible = summary[
        summary.selection_frequency.ge(0.50) & summary.sign_stability.ge(0.75)
    ].sort_values(
        ["selection_frequency", "sign_stability", "mean_abs_coefficient"],
        ascending=False,
    )
    selected: list[str] = []
    used_families: set[str] = set()
    for row in eligible.itertuples():
        if row.family in used_families:
            continue
        selected.append(row.candidate)
        used_families.add(row.family)
        if len(selected) == 4:
            break
    return selected, rows


def fit_final(
    train_parts: MatrixParts,
    valid_parts: MatrixParts,
    selected: list[str],
    y_train: pd.Series,
) -> tuple[np.ndarray, int, int]:
    train_matrix = train_parts.combine(selected)
    valid_matrix = valid_parts.combine(selected)
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
    result = probability, int(train_matrix.shape[1]), int(model.n_iter_.max())
    del train_matrix, valid_matrix, model
    return result


def run(
    output_dir: Path,
    n_repeats: int = 8,
    max_outer_folds: int = 8,
) -> dict[str, object]:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    X, families = build_zero_features(X_raw)
    candidates = list(families)
    unique_dates = X.TS.unique()
    splitter = KFold(n_splits=8, shuffle=True, random_state=0)
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
    trace_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    selected_by_fold: dict[int, list[str]] = {}
    for outer_fold, (train_date_idx, valid_date_idx) in enumerate(splitter.split(unique_dates), 1):
        if outer_fold > max_outer_folds:
            break
        train_dates = unique_dates[train_date_idx]
        valid_dates = unique_dates[valid_date_idx]
        train_mask = X.TS.isin(train_dates)
        valid_mask = X.TS.isin(valid_dates)
        design = ZeroFocusedDesign(candidates).fit(X.loc[train_mask])
        train_parts = design.transform(X.loc[train_mask])
        valid_parts = design.transform(X.loc[valid_mask])
        selected, trace = stability_select(
            X.loc[train_mask],
            y.loc[train_mask, "target_binarized"].astype(int),
            train_parts,
            candidates,
            families,
            outer_fold,
            n_repeats,
        )
        trace_rows.extend(trace)
        selected_by_fold[outer_fold] = selected
        probability, n_features, n_iter = fit_final(
            train_parts,
            valid_parts,
            selected,
            y.loc[train_mask, "target_binarized"].astype(int),
        )
        truth = y.loc[valid_mask, "target_binarized"].to_numpy(int)
        prediction = probability > 0.5
        oof.loc[valid_mask, "fold"] = outer_fold
        oof.loc[valid_mask, "score"] = probability
        oof.loc[valid_mask, "prediction"] = prediction.astype(int)
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_features": "|".join(selected),
                "n_selected": len(selected),
                "n_features": n_features,
                "n_iter": n_iter,
                "accuracy": accuracy_score(truth, prediction),
                "auc": roc_auc_score(truth, probability),
                "positive_rate": prediction.mean(),
            }
        )
        print(
            f"H7 OUTER={outer_fold} selected={selected} "
            f"accuracy={outer_rows[-1]['accuracy']:.6f}",
            flush=True,
        )
        del design, train_parts, valid_parts
        gc.collect()

    covered = oof.prediction.notna()
    oof.loc[covered, "is_correct"] = oof.loc[covered, "prediction"].astype(int).eq(
        oof.loc[covered, "y_true_binarized"].astype(int)
    ).astype(int)
    oof.index.name = "ROW_ID"
    trace = pd.DataFrame(trace_rows)
    trace.to_csv(output_dir / "stability_trace.csv", index=False)
    oof.to_csv(output_dir / "oof_zero_stability.csv")
    outer = pd.DataFrame(outer_rows)
    outer.to_csv(output_dir / "outer_fold_results.csv", index=False)

    frequency = (
        trace.groupby(["candidate", "family"], observed=True)
        .agg(
            selection_frequency=("selected_top8", "mean"),
            nonzero_frequency=("nonzero", "mean"),
            mean_abs_coefficient=("coefficient", lambda x: np.mean(np.abs(x))),
        )
        .reset_index()
        .sort_values("selection_frequency", ascending=False)
    )
    frequency.to_csv(output_dir / "selection_frequency.csv", index=False)

    candidate = oof.loc[covered]
    reference = pd.read_csv(REFERENCE_PATH, index_col="ROW_ID").loc[candidate.index]
    covered_dates = pd.Index(candidate.TS.unique())
    uncertainty = cluster_bootstrap_models(
        {"V2": reference, "H7 zero stability": candidate}, covered_dates, reference="V2"
    )
    uncertainty.to_csv(output_dir / "uncertainty.csv", index=False)
    ref_margin = reference.score.sub(0.5).abs()
    low_margin = ref_margin.le(ref_margin.median())
    low_margin_metrics = {
        "n_rows": int(low_margin.sum()),
        "v2_accuracy": float(reference.loc[low_margin, "is_correct"].mean()),
        "h7_accuracy": float(candidate.loc[low_margin, "is_correct"].mean()),
    }
    fold_compare = outer.set_index("outer_fold").join(
        reference.groupby("fold").is_correct.mean().rename("v2_accuracy")
    )
    fold_compare["gain_vs_v2"] = fold_compare.accuracy - fold_compare.v2_accuracy
    fold_compare.reset_index().to_csv(output_dir / "fold_comparison.csv", index=False)
    bootstrap_row = uncertainty[uncertainty.model.eq("H7 zero stability")].iloc[0]
    gate = {
        "positive_gain_folds": int(fold_compare.gain_vs_v2.gt(0).sum()),
        "bootstrap_gain_q025": float(bootstrap_row.gain_ci_low),
        "passed": bool(
            fold_compare.gain_vs_v2.gt(0).sum() >= 6 and bootstrap_row.gain_ci_low > 0
        ),
    }
    metrics = {
        "accuracy": float(candidate.is_correct.mean()),
        "auc": float(roc_auc_score(candidate.y_true_binarized, candidate.score)),
        "positive_rate": float(candidate.prediction.mean()),
        "v2_accuracy": float(reference.is_correct.mean()),
        "gain_vs_v2": float(candidate.is_correct.mean() - reference.is_correct.mean()),
        "low_margin": low_margin_metrics,
    }
    summary = {
        "n_candidates": len(candidates),
        "n_repeats": n_repeats,
        "max_outer_folds": max_outer_folds,
        "selection_C": SELECTION_C,
        "selection_l1_ratio": SELECTION_L1_RATIO,
        "selected_by_fold": selected_by_fold,
        "metrics": metrics,
        "gate": gate,
        "runtime_seconds": time.perf_counter() - start,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-repeats", type=int, default=8)
    parser.add_argument("--max-outer-folds", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_dir, n_repeats=args.n_repeats, max_outer_folds=args.max_outer_folds)

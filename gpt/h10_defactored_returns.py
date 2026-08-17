"""H10: test leakage-free market and GROUP defactoring of lagged returns."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from src.cross_validation import CVconfig, ModelConfig, run_cv  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h10_defactored_returns"
CATEGORICAL = ["ALLOCATION", "GROUP"]
MODEL_C = 0.003


def _leave_one_out_mean(
    frame: pd.DataFrame,
    groupers: list[pd.Series],
    columns: list[str],
) -> pd.DataFrame:
    """Return group means excluding the current row, without using y."""
    values = frame[columns].astype(float)
    grouped = frame.groupby(groupers, observed=True)[columns]
    sums = grouped.transform("sum")
    counts = grouped.transform("count")
    valid = values.notna().astype(int)
    denominator = (counts - valid).replace(0, np.nan)
    return (sums - values.fillna(0.0)).div(denominator)


def build_defactored_features(X: pd.DataFrame) -> pd.DataFrame:
    """Build a three-level return decomposition using contemporaneous X only."""
    result = X[["TS", "ALLOCATION", "GROUP"] + BASE_RETURNS].copy()
    date_mean = _leave_one_out_mean(result, [result["TS"]], BASE_RETURNS)
    group_mean = _leave_one_out_mean(
        result,
        [result["TS"], result["GROUP"]],
        BASE_RETURNS,
    )
    additions: dict[str, pd.Series] = {}
    for feature in BASE_RETURNS:
        additions[f"market_{feature}"] = date_mean[feature].astype("float32")
        additions[f"market_resid_{feature}"] = (
            result[feature] - date_mean[feature]
        ).astype("float32")
        additions[f"group_tilt_{feature}"] = (
            group_mean[feature] - date_mean[feature]
        ).astype("float32")
        additions[f"group_resid_{feature}"] = (
            result[feature] - group_mean[feature]
        ).astype("float32")
    return pd.concat(
        [result, pd.DataFrame(additions, index=result.index)],
        axis=1,
    )


MARKET_RESID = [f"market_resid_{feature}" for feature in BASE_RETURNS]
GROUP_RESID = [f"group_resid_{feature}" for feature in BASE_RETURNS]
MARKET_COMPONENT = [f"market_{feature}" for feature in BASE_RETURNS]
GROUP_TILT = [f"group_tilt_{feature}" for feature in BASE_RETURNS]

EXPERIMENTS: dict[str, dict[str, object]] = {
    "baseline_raw": {
        "numeric": BASE_RETURNS,
        "interaction": "RET_1",
    },
    "market_defactored_only": {
        "numeric": MARKET_RESID,
        "interaction": "market_resid_RET_1",
    },
    "group_defactored_only": {
        "numeric": GROUP_RESID,
        "interaction": "group_resid_RET_1",
    },
    "raw_plus_defactored": {
        "numeric": BASE_RETURNS + MARKET_RESID + GROUP_RESID,
        "interaction": "RET_1",
    },
    "hierarchical_components": {
        "numeric": MARKET_COMPONENT + GROUP_TILT + GROUP_RESID,
        "interaction": "group_resid_RET_1",
    },
    "hierarchical_raw_local_slope": {
        "numeric": MARKET_COMPONENT + GROUP_TILT + GROUP_RESID,
        "interaction": "RET_1",
    },
}


@dataclass
class DefactoredDesign:
    numeric_features: list[str]
    interaction_feature: str

    def __post_init__(self) -> None:
        self.numeric_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
        self.category_encoder = OneHotEncoder(handle_unknown="ignore")
        self.allocation_encoder = OneHotEncoder(handle_unknown="ignore")
        self.interaction_pipeline = None

    def fit(self, X: pd.DataFrame) -> "DefactoredDesign":
        self.numeric_pipeline.fit(X[self.numeric_features])
        self.category_encoder.fit(X[CATEGORICAL])
        self.allocation_encoder.fit(X[["ALLOCATION"]])
        if self.interaction_feature not in self.numeric_features:
            self.interaction_pipeline = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
            ).fit(X[[self.interaction_feature]])
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        numeric = self.numeric_pipeline.transform(X[self.numeric_features])
        categories = self.category_encoder.transform(X[CATEGORICAL])
        allocations = self.allocation_encoder.transform(X[["ALLOCATION"]])
        if self.interaction_feature in self.numeric_features:
            position = self.numeric_features.index(self.interaction_feature)
            interaction = numeric[:, position]
        else:
            interaction = self.interaction_pipeline.transform(
                X[[self.interaction_feature]]
            )[:, 0]
        allocation_slope = allocations.multiply(interaction[:, None])
        return sparse.hstack(
            [sparse.csr_matrix(numeric), categories, allocation_slope],
            format="csr",
        )


def make_preprocessor(
    numeric_features: list[str],
    interaction_feature: str,
) -> Callable[[pd.DataFrame, pd.DataFrame], tuple[sparse.csr_matrix, sparse.csr_matrix]]:
    def preprocess(
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        design = DefactoredDesign(numeric_features, interaction_feature).fit(X_train)
        return design.transform(X_train), design.transform(X_valid)

    return preprocess


def accuracy_summary(oof: pd.DataFrame) -> dict[str, float | int]:
    by_date = oof.groupby("TS", observed=True).is_correct.mean()
    return {
        "n_rows": int(len(oof)),
        "n_dates": int(len(by_date)),
        "accuracy": float(oof.is_correct.mean()),
        "date_standard_error": float(by_date.std(ddof=1) / np.sqrt(len(by_date))),
        "positive_prediction_rate": float(oof.prediction.mean()),
    }


def paired_uncertainty(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
) -> dict[str, float | int]:
    row_gain = challenger.is_correct.astype(float) - baseline.is_correct.astype(float)
    by_date = row_gain.groupby(baseline.TS).mean()
    by_fold = row_gain.groupby(baseline.fold).mean()
    se = float(by_date.std(ddof=1) / np.sqrt(len(by_date)))
    gain = float(row_gain.mean())
    return {
        "gain_accuracy": gain,
        "date_paired_standard_error": se,
        "ci95_low": gain - 1.96 * se,
        "ci95_high": gain + 1.96 * se,
        "folds_won": int((by_fold > 0).sum()),
        "folds_tied": int((by_fold == 0).sum()),
        "n_dates": int(len(by_date)),
    }


def component_diagnostics(X: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in BASE_RETURNS:
        raw_std = float(X[feature].std())
        for component in (
            f"market_{feature}",
            f"group_tilt_{feature}",
            f"group_resid_{feature}",
            f"market_resid_{feature}",
        ):
            rows.append(
                {
                    "lag": feature,
                    "component": component.removesuffix(f"_{feature}"),
                    "std": float(X[component].std()),
                    "std_over_raw": float(X[component].std() / raw_std),
                    "correlation_with_raw": float(X[[feature, component]].corr().iloc[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def run(
    names: list[str],
    output_dir: Path,
    n_splits: int = 8,
    random_state: int = 0,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    X = build_defactored_features(raw)
    component_diagnostics(X).to_csv(output_dir / "component_diagnostics.csv", index=False)

    oof_by_name: dict[str, pd.DataFrame] = {}
    result_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for name in names:
        specification = EXPERIMENTS[name]
        numeric = list(specification["numeric"])
        interaction = str(specification["interaction"])
        output = run_cv(
            X,
            y,
            ModelConfig(
                model=LogisticRegression(
                    C=MODEL_C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=200,
                    tol=1e-5,
                    random_state=0,
                ),
                regression=False,
                features=CATEGORICAL
                + numeric
                + ([interaction] if interaction not in numeric else []),
                preprocessing=make_preprocessor(numeric, interaction),
            ),
            CVconfig(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
                verbose=False,
            ),
        )
        oof = output.oof_results.copy()
        oof_by_name[name] = oof
        summary = accuracy_summary(oof)
        result_rows.append(
            {
                "experiment": name,
                "n_numeric_features": len(numeric),
                "interaction_feature": interaction,
                **summary,
            }
        )
        for fold in output.fold_results:
            fold_rows.append(
                {
                    "experiment": name,
                    "fold": fold.fold_id,
                    "n_train": int(np.asarray(fold.train_index).sum()),
                    "n_valid": int(np.asarray(fold.valid_index).sum()),
                    "train_accuracy": float(fold.train_metrics["accuracy"]),
                    "accuracy": float(fold.valid_metrics["accuracy"]),
                    "auc": float(fold.valid_metrics["auc"]),
                    "mcc": float(fold.valid_metrics["mcc"]),
                }
            )
        export = oof[
            [
                "fold",
                "TS",
                "ALLOCATION",
                "y_true_binarized",
                "score",
                "prediction",
                "is_correct",
            ]
        ].copy()
        export.index.name = "ROW_ID"
        export.to_csv(output_dir / f"oof_{name}.csv")
        pd.DataFrame(result_rows).to_csv(output_dir / "results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
        print(json.dumps(result_rows[-1], ensure_ascii=False))

    baseline = oof_by_name.get("baseline_raw")
    if baseline is None:
        raise ValueError("baseline_raw must be included for paired comparisons")
    uncertainty_rows = []
    for name, oof in oof_by_name.items():
        comparison = paired_uncertainty(baseline, oof)
        uncertainty_rows.append({"experiment": name, **comparison})
    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty.to_csv(output_dir / "paired_uncertainty.csv", index=False)

    results = pd.DataFrame(result_rows)
    folds = pd.DataFrame(fold_rows)
    merged = results.merge(
        uncertainty.rename(columns={"n_dates": "paired_n_dates"}),
        on="experiment",
        how="left",
    )
    eligible = merged[
        merged.experiment.ne("baseline_raw")
        & merged.gain_accuracy.gt(0)
        & merged.ci95_high.gt(0)
        & merged.folds_won.ge(max(3, int(np.ceil(n_splits * 5 / 8))))
    ]
    best = (
        eligible.sort_values("gain_accuracy", ascending=False).iloc[0].experiment
        if len(eligible)
        else None
    )
    summary: dict[str, object] = {
        "hypothesis": "H10 defactored returns",
        "model": "L2 logistic with fixed effects and allocation-specific RET_1 slope",
        "C": MODEL_C,
        "n_splits": n_splits,
        "random_state": random_state,
        "target_encoding": False,
        "date_order_used": False,
        "same_date_X_aggregation": True,
        "leave_one_out_aggregation": True,
        "best_eligible_experiment": best,
        "pca_authorized": best is not None,
        "results": merged.to_dict(orient="records"),
        "mean_fold_metrics": folds.groupby("experiment")[["train_accuracy", "accuracy", "auc", "mcc"]]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            run(
                names=arguments.experiments,
                output_dir=arguments.output_dir,
                n_splits=arguments.n_splits,
                random_state=arguments.random_state,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

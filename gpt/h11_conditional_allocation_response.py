"""H11: learn GROUP/allocation responses to compact X-only market states."""

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
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h11_conditional_allocation_response"
CATEGORICAL = ["ALLOCATION", "GROUP"]
MODEL_C = 0.003

STATE_FEATURES = [
    "market_ret1_mean",
    "market_ret1_dispersion",
    "market_ret1_breadth",
    "market_short_momentum",
    "market_recent_volatility",
    "group_ret1_relative",
    "group_momentum_relative",
    "group_dispersion_relative",
]


def _loo_mean(
    frame: pd.DataFrame,
    groupers: list[pd.Series],
    columns: list[str],
) -> pd.DataFrame:
    values = frame[columns].astype(float)
    grouped = frame.groupby(groupers, observed=True)[columns]
    sums = grouped.transform("sum")
    counts = grouped.transform("count")
    valid = values.notna().astype(int)
    return (sums - values.fillna(0.0)).div((counts - valid).replace(0, np.nan))


def _loo_std(
    frame: pd.DataFrame,
    groupers: list[pd.Series],
    column: str,
) -> pd.Series:
    values = frame[column].astype(float)
    valid = values.notna().astype(float)
    filled = values.fillna(0.0)
    helper = pd.DataFrame(
        {
            "value": filled,
            "square": filled.pow(2),
            "valid": valid,
        },
        index=frame.index,
    )
    grouped = helper.groupby(groupers, observed=True)
    sum_loo = grouped.value.transform("sum") - filled
    square_loo = grouped.square.transform("sum") - filled.pow(2)
    count_loo = grouped.valid.transform("sum") - valid
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = (square_loo - sum_loo.pow(2) / count_loo) / (count_loo - 1)
    return np.sqrt(variance.clip(lower=0)).where(count_loo > 1)


def build_state_features(X: pd.DataFrame) -> pd.DataFrame:
    result = X[
        ["TS", "ALLOCATION", "GROUP", "MEDIAN_DAILY_TURNOVER"] + BASE_RETURNS
    ].copy()
    recent = ["RET_1", "RET_2", "RET_3"]
    previous = ["RET_4", "RET_5", "RET_6", "RET_7", "RET_8", "RET_9", "RET_10"]
    vol_window = ["RET_1", "RET_2", "RET_3", "RET_4", "RET_5"]
    work = result.copy()
    work["row_short_momentum"] = X[recent].mean(axis=1) - X[previous].mean(axis=1)
    work["row_recent_volatility"] = X[vol_window].std(axis=1)
    work["row_ret1_positive"] = (X["RET_1"] > 0).astype(float)
    aggregate_columns = [
        "RET_1",
        "row_short_momentum",
        "row_recent_volatility",
        "row_ret1_positive",
    ]
    market_mean = _loo_mean(work, [work.TS], aggregate_columns)
    group_mean = _loo_mean(work, [work.TS, work.GROUP], aggregate_columns)
    market_dispersion = _loo_std(work, [work.TS], "RET_1")
    group_dispersion = _loo_std(work, [work.TS, work.GROUP], "RET_1")

    states = pd.DataFrame(
        {
            "market_ret1_mean": market_mean.RET_1,
            "market_ret1_dispersion": market_dispersion,
            "market_ret1_breadth": market_mean.row_ret1_positive,
            "market_short_momentum": market_mean.row_short_momentum,
            "market_recent_volatility": market_mean.row_recent_volatility,
            "group_ret1_relative": group_mean.RET_1 - market_mean.RET_1,
            "group_momentum_relative": (
                group_mean.row_short_momentum - market_mean.row_short_momentum
            ),
            "group_dispersion_relative": group_dispersion - market_dispersion,
        },
        index=X.index,
        dtype="float32",
    )
    result["row_short_momentum"] = work["row_short_momentum"].astype("float32")
    return pd.concat([result, states], axis=1)


EXPERIMENTS: dict[str, dict[str, object]] = {
    "baseline_raw": {
        "states": [],
        "group_scale": 0.0,
        "allocation_scale": 0.0,
        "style_scale": 0.0,
    },
    "state_main": {
        "states": STATE_FEATURES,
        "group_scale": 0.0,
        "allocation_scale": 0.0,
        "style_scale": 0.0,
    },
    "group_response": {
        "states": STATE_FEATURES,
        "group_scale": 1.0,
        "allocation_scale": 0.0,
        "style_scale": 0.0,
    },
    "allocation_response_strong": {
        "states": STATE_FEATURES,
        "group_scale": 0.0,
        "allocation_scale": 0.20,
        "style_scale": 0.0,
    },
    "allocation_response_very_strong": {
        "states": STATE_FEATURES,
        "group_scale": 0.0,
        "allocation_scale": 0.10,
        "style_scale": 0.0,
    },
    "allocation_response_moderate": {
        "states": STATE_FEATURES,
        "group_scale": 0.0,
        "allocation_scale": 0.35,
        "style_scale": 0.0,
    },
    "hierarchical_response": {
        "states": STATE_FEATURES,
        "group_scale": 0.50,
        "allocation_scale": 0.20,
        "style_scale": 0.0,
    },
    "style_low_rank_response": {
        "states": STATE_FEATURES,
        "group_scale": 0.0,
        "allocation_scale": 0.0,
        "style_scale": 0.50,
    },
}


PROFILE_FEATURES = [
    "ret1_mean",
    "ret1_std",
    "ret1_abs_mean",
    "ret1_ret2_corr",
    "ret1_market_corr",
    "ret1_group_state_corr",
    "short_momentum_mean",
    "turnover_median",
]


def allocation_style_profile(X: pd.DataFrame) -> pd.DataFrame:
    """Continuous X-only style descriptors learned on training dates only."""
    grouped = X.groupby("ALLOCATION", observed=True)
    profile = grouped.RET_1.agg(ret1_mean="mean", ret1_std="std")
    profile["ret1_abs_mean"] = X.RET_1.abs().groupby(X.ALLOCATION).mean()
    profile["ret1_ret2_corr"] = grouped.apply(
        lambda block: block.RET_1.corr(block.RET_2),
        include_groups=False,
    )
    profile["ret1_market_corr"] = grouped.apply(
        lambda block: block.RET_1.corr(block.market_ret1_mean),
        include_groups=False,
    )
    profile["ret1_group_state_corr"] = grouped.apply(
        lambda block: block.RET_1.corr(block.group_ret1_relative),
        include_groups=False,
    )
    profile["short_momentum_mean"] = grouped.row_short_momentum.mean()
    profile["turnover_median"] = grouped.MEDIAN_DAILY_TURNOVER.median()
    return profile[PROFILE_FEATURES].sort_index()


@dataclass
class ConditionalResponseDesign:
    state_features: list[str]
    group_scale: float
    allocation_scale: float
    style_scale: float

    def __post_init__(self) -> None:
        self.numeric_features = BASE_RETURNS + self.state_features
        self.numeric_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
        self.category_encoder = OneHotEncoder(handle_unknown="ignore")
        self.group_encoder = OneHotEncoder(handle_unknown="ignore")
        self.allocation_encoder = OneHotEncoder(handle_unknown="ignore")
        self.profile_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
        self.profile_coordinates = None

    def fit(self, X: pd.DataFrame) -> "ConditionalResponseDesign":
        self.numeric_pipeline.fit(X[self.numeric_features])
        self.category_encoder.fit(X[CATEGORICAL])
        self.group_encoder.fit(X[["GROUP"]])
        self.allocation_encoder.fit(X[["ALLOCATION"]])
        if self.style_scale > 0:
            profiles = allocation_style_profile(X)
            coordinates = self.profile_pipeline.fit_transform(profiles)
            self.profile_coordinates = pd.DataFrame(
                coordinates,
                index=profiles.index.astype(str),
                columns=PROFILE_FEATURES,
            )
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        numeric = self.numeric_pipeline.transform(X[self.numeric_features])
        categories = self.category_encoder.transform(X[CATEGORICAL])
        allocations = self.allocation_encoder.transform(X[["ALLOCATION"]])
        blocks: list[sparse.spmatrix] = [sparse.csr_matrix(numeric), categories]

        ret1 = numeric[:, self.numeric_features.index("RET_1")]
        blocks.append(allocations.multiply(ret1[:, None]))

        if self.state_features:
            state_positions = [self.numeric_features.index(name) for name in self.state_features]
            states = numeric[:, state_positions]
            if self.group_scale > 0:
                groups = self.group_encoder.transform(X[["GROUP"]])
                group_blocks = [
                    groups.multiply(states[:, position, None]) * self.group_scale
                    for position in range(states.shape[1])
                ]
                blocks.extend(group_blocks)
            if self.allocation_scale > 0:
                allocation_blocks = [
                    allocations.multiply(states[:, position, None]) * self.allocation_scale
                    for position in range(states.shape[1])
                ]
                blocks.extend(allocation_blocks)
            if self.style_scale > 0:
                allocation_labels = X.ALLOCATION.astype(str).to_numpy()
                profiles = self.profile_coordinates.reindex(allocation_labels).to_numpy(float)
                profiles = np.nan_to_num(profiles, nan=0.0)
                style_blocks = [
                    sparse.csr_matrix(profiles * states[:, position, None]) * self.style_scale
                    for position in range(states.shape[1])
                ]
                blocks.extend(style_blocks)
        return sparse.hstack(blocks, format="csr")


def make_preprocessor(
    state_features: list[str],
    group_scale: float,
    allocation_scale: float,
    style_scale: float,
) -> Callable[[pd.DataFrame, pd.DataFrame], tuple[sparse.csr_matrix, sparse.csr_matrix]]:
    def preprocess(
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        design = ConditionalResponseDesign(
            state_features,
            group_scale,
            allocation_scale,
            style_scale,
        ).fit(X_train)
        return design.transform(X_train), design.transform(X_valid)

    return preprocess


def paired_uncertainty(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
) -> dict[str, float | int]:
    row_gain = challenger.is_correct.astype(float) - baseline.is_correct.astype(float)
    date_gain = row_gain.groupby(baseline.TS).mean()
    fold_gain = row_gain.groupby(baseline.fold).mean()
    gain = float(row_gain.mean())
    se = float(date_gain.std(ddof=1) / np.sqrt(len(date_gain)))
    return {
        "gain_accuracy": gain,
        "date_paired_standard_error": se,
        "ci95_low": gain - 1.96 * se,
        "ci95_high": gain + 1.96 * se,
        "folds_won": int((fold_gain > 0).sum()),
        "folds_tied": int((fold_gain == 0).sum()),
        "paired_n_dates": int(len(date_gain)),
    }


def state_diagnostics(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in STATE_FEATURES:
        by_date = X.groupby("TS", observed=True)[feature].mean()
        rows.append(
            {
                "feature": feature,
                "row_mean": float(X[feature].mean()),
                "row_std": float(X[feature].std()),
                "date_mean_std": float(by_date.std()),
                "missing_rate": float(X[feature].isna().mean()),
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
    X = build_state_features(raw)
    state_diagnostics(X).to_csv(output_dir / "state_diagnostics.csv", index=False)

    oof_by_name: dict[str, pd.DataFrame] = {}
    result_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for name in names:
        specification = EXPERIMENTS[name]
        states = list(specification["states"])
        group_scale = float(specification["group_scale"])
        allocation_scale = float(specification["allocation_scale"])
        style_scale = float(specification["style_scale"])
        numeric = BASE_RETURNS + states
        output = run_cv(
            X,
            y,
            ModelConfig(
                model=LogisticRegression(
                    C=MODEL_C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=250,
                    tol=1e-5,
                    random_state=0,
                ),
                regression=False,
                features=CATEGORICAL
                + numeric
                + ["MEDIAN_DAILY_TURNOVER", "row_short_momentum"],
                preprocessing=make_preprocessor(
                    states,
                    group_scale,
                    allocation_scale,
                    style_scale,
                ),
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
        by_date = oof.groupby("TS", observed=True).is_correct.mean()
        result_rows.append(
            {
                "experiment": name,
                "n_states": len(states),
                "group_scale": group_scale,
                "allocation_scale": allocation_scale,
                "style_scale": style_scale,
                "accuracy": float(oof.is_correct.mean()),
                "date_standard_error": float(by_date.std(ddof=1) / np.sqrt(len(by_date))),
                "positive_prediction_rate": float(oof.prediction.mean()),
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
                    "train_auc": float(fold.train_metrics["auc"]),
                    "auc": float(fold.valid_metrics["auc"]),
                    "mcc": float(fold.valid_metrics["mcc"]),
                }
            )
        export = oof[
            ["fold", "TS", "ALLOCATION", "y_true_binarized", "score", "prediction", "is_correct"]
        ].copy()
        export.index.name = "ROW_ID"
        export.to_csv(output_dir / f"oof_{name}.csv")
        pd.DataFrame(result_rows).to_csv(output_dir / "results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
        print(json.dumps(result_rows[-1], ensure_ascii=False))

    if "baseline_raw" not in oof_by_name:
        raise ValueError("baseline_raw must be included")
    baseline = oof_by_name["baseline_raw"]
    uncertainty = pd.DataFrame(
        [
            {"experiment": name, **paired_uncertainty(baseline, oof)}
            for name, oof in oof_by_name.items()
        ]
    )
    uncertainty.to_csv(output_dir / "paired_uncertainty.csv", index=False)
    results = pd.DataFrame(result_rows).merge(uncertainty, on="experiment", how="left")
    folds = pd.DataFrame(fold_rows)
    challengers = results[results.experiment.ne("baseline_raw")]
    eligible = challengers[
        challengers.gain_accuracy.gt(0)
        & challengers.folds_won.ge(max(3, int(np.ceil(n_splits * 5 / 8))))
        & challengers.ci95_high.gt(0)
    ]
    best = (
        eligible.sort_values("gain_accuracy", ascending=False).iloc[0].experiment
        if len(eligible)
        else None
    )
    summary: dict[str, object] = {
        "hypothesis": "H11 conditional allocation response",
        "model_C": MODEL_C,
        "n_splits": n_splits,
        "target_encoding": False,
        "date_order_used": False,
        "same_date_X_aggregation": True,
        "leave_one_out_aggregation": True,
        "best_eligible_experiment": best,
        "low_rank_followup_authorized": best is not None,
        "results": results.to_dict(orient="records"),
        "mean_fold_metrics": folds.groupby("experiment")[
            ["train_accuracy", "accuracy", "train_auc", "auc", "mcc"]
        ].mean().reset_index().to_dict(orient="records"),
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
                arguments.experiments,
                arguments.output_dir,
                arguments.n_splits,
                arguments.random_state,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

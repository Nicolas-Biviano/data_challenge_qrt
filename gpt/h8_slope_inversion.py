"""H8: reliability and X-only predictability of date-level slope inversions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.adversarial_stress_validation import date_profile  # noqa: E402
from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h8_slope_inversion"


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 8 or np.std(x[valid]) <= 1e-12 or np.std(y[valid]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def fisher(correlation: float) -> float:
    return float(np.arctanh(np.clip(correlation, -0.999, 0.999)))


def compute_date_slopes(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    split_hash = pd.util.hash_pandas_object(X.ALLOCATION.astype(str), index=False).to_numpy()
    half = (split_hash % 2).astype(int)
    rows: list[dict[str, object]] = []
    for date, positions in X.groupby("TS", observed=True).indices.items():
        positions = np.asarray(positions)
        target = y.iloc[positions].to_numpy(float)
        local_half = half[positions]
        for feature in BASE_RETURNS:
            values = X.iloc[positions][feature].to_numpy(float)
            corr = safe_corr(values, target)
            mask_0 = local_half == 0
            mask_1 = local_half == 1
            corr_0 = safe_corr(values[mask_0], target[mask_0])
            corr_1 = safe_corr(values[mask_1], target[mask_1])
            if not np.isfinite(corr):
                continue
            rows.append(
                {
                    "TS": str(date),
                    "feature": feature,
                    "n_rows": int(np.isfinite(values).sum()),
                    "correlation": corr,
                    "fisher_z": fisher(corr),
                    "sampling_variance": 1.0 / max(np.isfinite(values).sum() - 3, 1),
                    "n_half_0": int(mask_0.sum()),
                    "n_half_1": int(mask_1.sum()),
                    "corr_half_0": corr_0,
                    "corr_half_1": corr_1,
                    "z_half_0": fisher(corr_0) if np.isfinite(corr_0) else np.nan,
                    "z_half_1": fisher(corr_1) if np.isfinite(corr_1) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def random_effect_parameters(values: pd.DataFrame) -> tuple[float, float]:
    z = values.fisher_z.to_numpy(float)
    sampling = values.sampling_variance.to_numpy(float)
    weights = 1.0 / sampling
    global_mean = float(np.average(z, weights=weights))
    observed_variance = float(np.var(z, ddof=1))
    tau2 = max(observed_variance - float(np.mean(sampling)), 0.0)
    return global_mean, tau2


def apply_shrinkage(values: pd.DataFrame, global_mean: float, tau2: float) -> np.ndarray:
    sampling = values.sampling_variance.to_numpy(float)
    if tau2 <= 0:
        return np.full(len(values), global_mean)
    weight = tau2 / (tau2 + sampling)
    return global_mean + weight * (values.fisher_z.to_numpy(float) - global_mean)


def reliability_table(slopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, block in slopes.groupby("feature", observed=True):
        valid = block[["z_half_0", "z_half_1"]].dropna()
        se0 = 1.0 / np.sqrt(np.maximum(block.n_half_0 - 3, 1))
        se1 = 1.0 / np.sqrt(np.maximum(block.n_half_1 - 3, 1))
        reliable = block.z_half_0.abs().gt(se0) & block.z_half_1.abs().gt(se1)
        global_mean, tau2 = random_effect_parameters(block)
        shrunk = apply_shrinkage(block, global_mean, tau2)
        sign_agreement = np.mean(np.sign(valid.z_half_0) == np.sign(valid.z_half_1))
        reliable_agreement = (
            np.mean(np.sign(block.loc[reliable, "z_half_0"]) == np.sign(block.loc[reliable, "z_half_1"]))
            if reliable.sum() else np.nan
        )
        rows.append(
            {
                "feature": feature,
                "n_dates": len(block),
                "global_fisher_z": global_mean,
                "global_correlation": float(np.tanh(global_mean)),
                "between_date_variance_tau2": tau2,
                "split_half_pearson": pearsonr(valid.z_half_0, valid.z_half_1).statistic,
                "split_half_spearman": spearmanr(valid.z_half_0, valid.z_half_1).statistic,
                "split_half_sign_agreement": sign_agreement,
                "n_reliable_both_halves": int(reliable.sum()),
                "reliable_sign_agreement": reliable_agreement,
                "shrunk_inversion_rate": float(np.mean(shrunk * global_mean < 0)),
                "reproducibility_gate": bool(
                    pearsonr(valid.z_half_0, valid.z_half_1).statistic > 0
                    and sign_agreement > 0.52
                ),
            }
        )
    return pd.DataFrame(rows)


def build_target_matrix(
    slopes: pd.DataFrame,
    dates: pd.Index,
    train_dates: pd.Index,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    parameters: dict[str, tuple[float, float]] = {}
    columns = {}
    for feature in BASE_RETURNS:
        block = slopes[slopes.feature.eq(feature)].set_index("TS").loc[dates]
        train_block = block.loc[train_dates]
        global_mean, tau2 = random_effect_parameters(train_block)
        parameters[feature] = (global_mean, tau2)
        columns[feature] = pd.Series(
            apply_shrinkage(block, global_mean, tau2), index=dates
        )
    return pd.DataFrame(columns, index=dates), parameters


def fit_predict_slopes(
    profiles: pd.DataFrame,
    slopes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = profiles.index
    splitter = KFold(n_splits=8, shuffle=True, random_state=0)
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for fold, (train_position, valid_position) in enumerate(splitter.split(dates), 1):
        train_dates = dates[train_position]
        valid_dates = dates[valid_position]
        targets, parameters = build_target_matrix(slopes, dates, train_dates)
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n_components=min(15, profiles.shape[1]), random_state=0),
            Ridge(alpha=100.0),
        )
        pipeline.fit(profiles.loc[train_dates], targets.loc[train_dates])
        predicted = pd.DataFrame(
            pipeline.predict(profiles.loc[valid_dates]),
            index=valid_dates,
            columns=BASE_RETURNS,
        )
        actual = targets.loc[valid_dates]
        for feature in BASE_RETURNS:
            global_mean, tau2 = parameters[feature]
            feature_block = slopes[slopes.feature.eq(feature)].set_index("TS").loc[valid_dates]
            actual_values = actual[feature].to_numpy(float)
            predicted_values = predicted[feature].to_numpy(float)
            inversion = actual_values * global_mean < 0
            predicted_inversion = predicted_values * global_mean < 0
            inversion_score = -predicted_values * np.sign(global_mean)
            auc = (
                roc_auc_score(inversion.astype(int), inversion_score)
                if np.unique(inversion).size == 2 else np.nan
            )
            balanced = (
                balanced_accuracy_score(inversion, predicted_inversion)
                if np.unique(inversion).size == 2 else np.nan
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "n_dates": len(valid_dates),
                    "global_fisher_z": global_mean,
                    "tau2": tau2,
                    "correlation": safe_corr(predicted_values, actual_values),
                    "r2": r2_score(actual_values, predicted_values),
                    "mae_model": mean_absolute_error(actual_values, predicted_values),
                    "mae_constant": mean_absolute_error(
                        actual_values, np.full(len(actual_values), global_mean)
                    ),
                    "inversion_rate": inversion.mean(),
                    "inversion_auc": auc,
                    "inversion_balanced_accuracy": balanced,
                }
            )
            for date, raw_z, actual_z, predicted_z, is_inversion in zip(
                valid_dates,
                feature_block.fisher_z,
                actual_values,
                predicted_values,
                inversion,
            ):
                prediction_rows.append(
                    {
                        "TS": date,
                        "fold": fold,
                        "feature": feature,
                        "raw_fisher_z": raw_z,
                        "actual_shrunk_z": actual_z,
                        "predicted_shrunk_z": predicted_z,
                        "global_fisher_z": global_mean,
                        "inversion": bool(is_inversion),
                    }
                )
    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows)


def summarize_prediction(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    reliability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for feature, block in predictions.groupby("feature", observed=True):
        inversion = block.inversion.astype(int)
        inversion_score = -block.predicted_shrunk_z * np.sign(block.global_fisher_z)
        predicted_inversion = block.predicted_shrunk_z * block.global_fisher_z < 0
        feature_folds = fold_metrics[fold_metrics.feature.eq(feature)]
        rows.append(
            {
                "feature": feature,
                "n_dates": len(block),
                "correlation": safe_corr(
                    block.predicted_shrunk_z.to_numpy(), block.actual_shrunk_z.to_numpy()
                ),
                "r2": r2_score(block.actual_shrunk_z, block.predicted_shrunk_z),
                "mae_model": mean_absolute_error(block.actual_shrunk_z, block.predicted_shrunk_z),
                "mae_constant": mean_absolute_error(block.actual_shrunk_z, block.global_fisher_z),
                "mae_improvement_folds": int(
                    feature_folds.mae_model.lt(feature_folds.mae_constant).sum()
                ),
                "inversion_rate": inversion.mean(),
                "inversion_auc": (
                    roc_auc_score(inversion, inversion_score)
                    if inversion.nunique() == 2 else np.nan
                ),
                "inversion_balanced_accuracy": (
                    balanced_accuracy_score(inversion, predicted_inversion)
                    if inversion.nunique() == 2 else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows).merge(
        reliability[["feature", "reproducibility_gate"]], on="feature", how="left"
    )
    gate = summary.assign(
        gate_passed=lambda d: (
            d.reproducibility_gate
            & d.correlation.gt(0.10)
            & d.inversion_auc.gt(0.55)
            & d.mae_improvement_folds.ge(6)
        )
    )
    return summary, gate[[
        "feature", "reproducibility_gate", "correlation", "inversion_auc",
        "mae_improvement_folds", "gate_passed"
    ]]


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train().target
    if not X.index.equals(y.index):
        raise ValueError("X_train et y_train ne sont pas alignés par ROW_ID")
    slopes = compute_date_slopes(X, y)
    reliability = reliability_table(slopes)
    profiles = date_profile(X)
    profiles.index = profiles.index.astype(str)
    common_dates = profiles.index.intersection(slopes.TS.unique())
    profiles = profiles.loc[common_dates]
    predictions, fold_metrics = fit_predict_slopes(profiles, slopes)
    feature_metrics, gate = summarize_prediction(predictions, fold_metrics, reliability)

    slopes.to_csv(output_dir / "date_slopes.csv", index=False)
    reliability.to_csv(output_dir / "split_half_reliability.csv", index=False)
    profiles.to_csv(output_dir / "date_profiles_x_only.csv")
    predictions.to_csv(output_dir / "oof_slope_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    feature_metrics.to_csv(output_dir / "feature_metrics.csv", index=False)
    gate.to_csv(output_dir / "gate.csv", index=False)
    summary = {
        "n_dates": len(profiles),
        "features": BASE_RETURNS,
        "n_reproducible_features": int(reliability.reproducibility_gate.sum()),
        "n_gate_passed": int(gate.gate_passed.sum()),
        "reliability": reliability.to_dict("records"),
        "prediction_metrics": feature_metrics.to_dict("records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print("\nRELIABILITY")
    print(reliability.to_string(index=False))
    print("\nPREDICTION")
    print(feature_metrics.to_string(index=False))
    print("\nGATE")
    print(gate.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_dir)

"""H6: strict cross-fitted conditional error probabilities and model selectors."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h6_error_estimator"
V1_PATH = GPT_DIR / "outputs" / "v1_recheck" / "oof_predictions.csv"
LGBM_PATH = (
    GPT_DIR
    / "outputs"
    / "v3_lgbm_linear_full8"
    / "oof_lgbm_linear_returns_compact_full.csv"
)
PANEL_MEMBERSHIPS = GPT_DIR / "outputs" / "test_sized_panels" / "panel_memberships.csv"

RETURN_LAGS = [1, 2, 3, 4, 7, 8, 9, 18]
CATEGORICAL_COLUMNS = ["ALLOCATION", "GROUP"]
ERROR_MODELS = ["v1", "lgbm"]
ESTIMATORS = ["constant", "amplitude_logit", "compact_logit"]
SELECTORS = [
    "V1",
    "LightGBM",
    "amplitude_rank",
    "lower_error_amplitude",
    "lower_error_compact",
    "direct_compact",
]


def load_base_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    X = ChallengeDataLoader.load_X_train()
    v1 = pd.read_csv(V1_PATH, index_col="ROW_ID")
    lgbm = pd.read_csv(LGBM_PATH, index_col="ROW_ID")
    common = X.index.intersection(v1.index).intersection(lgbm.index)
    if len(common) != len(X):
        raise ValueError(f"OOF alignment incomplete: {len(common):,}/{len(X):,} rows")
    if not np.array_equal(
        v1.loc[common, "y_true_binarized"].astype(int).to_numpy(),
        lgbm.loc[common, "y_true_binarized"].astype(int).to_numpy(),
    ):
        raise ValueError("V1 and LightGBM truths do not align")
    if not np.array_equal(
        v1.loc[common, "fold"].astype(int).to_numpy(),
        lgbm.loc[common, "fold"].astype(int).to_numpy(),
    ):
        raise ValueError("V1 and LightGBM folds do not align")

    meta = pd.DataFrame(index=common)
    meta["fold"] = v1.loc[common, "fold"].astype(int)
    meta["TS"] = X.loc[common, "TS"].astype(str)
    meta["truth"] = v1.loc[common, "y_true_binarized"].astype(int)
    meta["v1_score"] = v1.loc[common, "score"].astype(float)
    meta["lgbm_score"] = lgbm.loc[common, "score"].astype(float)
    meta["v1_prediction"] = v1.loc[common, "prediction"].astype(int)
    meta["lgbm_prediction"] = lgbm.loc[common, "prediction"].astype(int)
    meta["error_v1"] = meta.v1_prediction.ne(meta.truth).astype(int)
    meta["error_lgbm"] = meta.lgbm_prediction.ne(meta.truth).astype(int)
    meta["disagreement"] = meta.v1_prediction.ne(meta.lgbm_prediction).astype(int)

    features, numeric_columns = build_features(X.loc[common], meta)
    return meta, features, numeric_columns


def build_features(X: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = pd.DataFrame(index=X.index)
    features["v1_score"] = meta.v1_score
    features["lgbm_score"] = meta.lgbm_score
    features["abs_v1_score"] = meta.v1_score.abs()
    features["abs_lgbm_score"] = meta.lgbm_score.abs()
    features["mean_score"] = (meta.v1_score + meta.lgbm_score) / 2
    features["score_difference"] = meta.lgbm_score - meta.v1_score
    features["abs_score_difference"] = features.score_difference.abs()
    features["min_abs_score"] = np.minimum(features.abs_v1_score, features.abs_lgbm_score)
    features["max_abs_score"] = np.maximum(features.abs_v1_score, features.abs_lgbm_score)
    features["score_product"] = meta.v1_score * meta.lgbm_score
    features["disagreement"] = meta.disagreement
    features["v1_date_confidence"] = features.groupby(meta.TS).abs_v1_score.rank(pct=True)
    features["lgbm_date_confidence"] = features.groupby(meta.TS).abs_lgbm_score.rank(pct=True)

    for lag in RETURN_LAGS:
        column = f"RET_{lag}"
        features[column] = X[column]
        features[f"abs_{column}"] = X[column].abs()

    all_returns = X[[f"RET_{lag}" for lag in range(1, 21)]]
    recent_returns = X[[f"RET_{lag}" for lag in range(1, 5)]]
    features["return_recent_mean"] = recent_returns.mean(axis=1)
    features["return_recent_std"] = recent_returns.std(axis=1)
    features["return_all_mean"] = all_returns.mean(axis=1)
    features["return_all_std"] = all_returns.std(axis=1)
    features["return_recent_minus_old"] = (
        recent_returns.mean(axis=1)
        - X[[f"RET_{lag}" for lag in range(5, 21)]].mean(axis=1)
    )
    signs = np.sign(all_returns.fillna(0).to_numpy())
    features["return_sign_changes"] = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)

    volume_columns = [f"SIGNED_VOLUME_{lag}" for lag in range(1, 21)]
    volumes = X[volume_columns]
    features["volume_observed_fraction"] = volumes.notna().mean(axis=1)
    features["sv1_observed"] = X.SIGNED_VOLUME_1.notna().astype(int)
    features["signed_volume_1"] = X.SIGNED_VOLUME_1
    features["abs_signed_volume_1"] = X.SIGNED_VOLUME_1.abs()
    features["mean_abs_observed_volume"] = volumes.abs().mean(axis=1)
    features["median_daily_turnover"] = X.MEDIAN_DAILY_TURNOVER
    features["log1p_turnover"] = np.log1p(X.MEDIAN_DAILY_TURNOVER.clip(lower=0))

    by_date = X.assign(_ret1_abs=X.RET_1.abs(), _sv1_missing=X.SIGNED_VOLUME_1.isna()).groupby(
        "TS", observed=True
    )
    features["date_n_rows"] = by_date.RET_1.transform("size")
    features["date_ret1_mean"] = by_date.RET_1.transform("mean")
    features["date_ret1_std"] = by_date.RET_1.transform("std")
    features["date_ret1_abs_mean"] = by_date._ret1_abs.transform("mean")
    features["date_ret1_positive_rate"] = X.RET_1.gt(0).groupby(X.TS).transform("mean")
    features["date_sv1_missing_rate"] = by_date._sv1_missing.transform("mean")
    features["date_turnover_median"] = by_date.MEDIAN_DAILY_TURNOVER.transform("median")
    features["date_abs_ret1_rank"] = X.RET_1.abs().groupby(X.TS).rank(pct=True)
    features["date_turnover_rank"] = X.MEDIAN_DAILY_TURNOVER.groupby(X.TS).rank(pct=True)

    features["ALLOCATION"] = X.ALLOCATION.astype(str)
    features["GROUP"] = X.GROUP.astype(str)
    numeric_columns = [c for c in features if c not in CATEGORICAL_COLUMNS]
    features[numeric_columns] = features[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return features, numeric_columns


def compact_preprocessor(numeric_columns: list[str]) -> ColumnTransformer:
    numeric = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("categorical", categorical, CATEGORICAL_COLUMNS)],
        sparse_threshold=0.3,
    )


def logistic() -> LogisticRegression:
    return LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=300,
        tol=1e-5,
        random_state=0,
    )


def probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=int)
    return {
        "n_rows": int(len(y)),
        "event_rate": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probability)) if np.unique(y).size > 1 else np.nan,
    }


def record_metric(
    rows: list[dict[str, object]],
    fold: int,
    base_model: str,
    estimator: str,
    split: str,
    y: np.ndarray,
    probability: np.ndarray,
) -> None:
    rows.append(
        {
            "fold": fold,
            "base_model": base_model,
            "estimator": estimator,
            "split": split,
            **probability_metrics(y, probability),
        }
    )


def fit_cross_fitted(
    meta: pd.DataFrame,
    features: pd.DataFrame,
    numeric_columns: list[str],
    fold_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = sorted(meta.fold.unique())
    if fold_limit is not None:
        folds = folds[:fold_limit]
    predictions = pd.DataFrame(index=meta.index)
    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    for fold in folds:
        train_mask = meta.fold.ne(fold)
        valid_mask = meta.fold.eq(fold)
        train_index = meta.index[train_mask]
        valid_index = meta.index[valid_mask]
        print(f"fold {fold}: train={train_mask.sum():,}, valid={valid_mask.sum():,}", flush=True)

        preprocessor = compact_preprocessor(numeric_columns)
        X_train_compact = preprocessor.fit_transform(features.loc[train_index])
        X_valid_compact = preprocessor.transform(features.loc[valid_index])
        transformed_names = preprocessor.get_feature_names_out()

        for base_model in ERROR_MODELS:
            target_column = f"error_{base_model}"
            y_train = meta.loc[train_index, target_column].to_numpy(dtype=int)
            y_valid = meta.loc[valid_index, target_column].to_numpy(dtype=int)

            constant_probability = np.full(len(valid_index), y_train.mean())
            predictions.loc[valid_index, f"p_error_{base_model}_constant"] = constant_probability
            record_metric(
                metric_rows, fold, base_model, "constant", "valid",
                y_valid, constant_probability,
            )

            amplitude_columns = [f"abs_{base_model}_score", f"{base_model}_date_confidence"]
            amplitude_model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                logistic(),
            )
            amplitude_model.fit(features.loc[train_index, amplitude_columns], y_train)
            p_train_amplitude = amplitude_model.predict_proba(
                features.loc[train_index, amplitude_columns]
            )[:, 1]
            p_valid_amplitude = amplitude_model.predict_proba(
                features.loc[valid_index, amplitude_columns]
            )[:, 1]
            predictions.loc[valid_index, f"p_error_{base_model}_amplitude_logit"] = p_valid_amplitude
            record_metric(
                metric_rows, fold, base_model, "amplitude_logit", "train",
                y_train, p_train_amplitude,
            )
            record_metric(
                metric_rows, fold, base_model, "amplitude_logit", "valid",
                y_valid, p_valid_amplitude,
            )

            compact_model = logistic()
            compact_model.fit(X_train_compact, y_train)
            p_train_compact = compact_model.predict_proba(X_train_compact)[:, 1]
            p_valid_compact = compact_model.predict_proba(X_valid_compact)[:, 1]
            predictions.loc[valid_index, f"p_error_{base_model}_compact_logit"] = p_valid_compact
            record_metric(
                metric_rows, fold, base_model, "compact_logit", "train",
                y_train, p_train_compact,
            )
            record_metric(
                metric_rows, fold, base_model, "compact_logit", "valid",
                y_valid, p_valid_compact,
            )
            for feature_name, coefficient in zip(transformed_names, compact_model.coef_[0]):
                coefficient_rows.append(
                    {
                        "fold": fold,
                        "target": f"error_{base_model}",
                        "feature": feature_name,
                        "coefficient": coefficient,
                    }
                )

        train_disagreement = train_mask & meta.disagreement.eq(1)
        valid_disagreement = valid_mask & meta.disagreement.eq(1)
        train_positions = np.flatnonzero(meta.loc[train_index, "disagreement"].to_numpy() == 1)
        valid_positions = np.flatnonzero(meta.loc[valid_index, "disagreement"].to_numpy() == 1)
        y_choose_train = meta.loc[meta.index[train_disagreement], "error_v1"].to_numpy(dtype=int)
        direct_model = logistic()
        direct_model.fit(X_train_compact[train_positions], y_choose_train)
        p_direct_train = direct_model.predict_proba(X_train_compact[train_positions])[:, 1]
        p_direct_valid = direct_model.predict_proba(X_valid_compact[valid_positions])[:, 1]
        predictions.loc[meta.index[valid_disagreement], "p_lgbm_correct_direct_compact"] = p_direct_valid
        record_metric(
            metric_rows, fold, "selector", "direct_compact", "train",
            y_choose_train, p_direct_train,
        )
        y_choose_valid = meta.loc[meta.index[valid_disagreement], "error_v1"].to_numpy(dtype=int)
        record_metric(
            metric_rows, fold, "selector", "direct_compact", "valid",
            y_choose_valid, p_direct_valid,
        )
        for feature_name, coefficient in zip(transformed_names, direct_model.coef_[0]):
            coefficient_rows.append(
                {
                    "fold": fold,
                    "target": "choose_lgbm_on_disagreement",
                    "feature": feature_name,
                    "coefficient": coefficient,
                }
            )

        del X_train_compact, X_valid_compact, preprocessor
        gc.collect()

    predictions = predictions.loc[meta.fold.isin(folds)].copy()
    return predictions, pd.DataFrame(metric_rows), pd.DataFrame(coefficient_rows)


def calibration_table(meta: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    aligned = meta.loc[predictions.index]
    for base_model in ERROR_MODELS:
        y = aligned[f"error_{base_model}"]
        for estimator in ESTIMATORS:
            probability = predictions[f"p_error_{base_model}_{estimator}"]
            decile = np.minimum((probability.rank(method="first", pct=True) * 10).astype(int), 9)
            table = pd.DataFrame({"y": y, "p": probability, "decile": decile})
            for value, block in table.groupby("decile"):
                rows.append(
                    {
                        "base_model": base_model,
                        "estimator": estimator,
                        "predicted_error_decile": int(value) + 1,
                        "n_rows": len(block),
                        "mean_predicted_error": block.p.mean(),
                        "actual_error_rate": block.y.mean(),
                    }
                )
    return pd.DataFrame(rows)


def overall_error_metrics(meta: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    aligned = meta.loc[predictions.index]
    for base_model in ERROR_MODELS:
        y = aligned[f"error_{base_model}"].to_numpy()
        for estimator in ESTIMATORS:
            probability = predictions[f"p_error_{base_model}_{estimator}"].to_numpy()
            rows.append(
                {
                    "base_model": base_model,
                    "estimator": estimator,
                    **probability_metrics(y, probability),
                }
            )
    return pd.DataFrame(rows)


def build_selectors(meta: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    aligned = meta.loc[predictions.index]
    selectors = pd.DataFrame(index=predictions.index)
    selectors["fold"] = aligned.fold
    selectors["TS"] = aligned.TS
    selectors["truth"] = aligned.truth
    selectors["disagreement"] = aligned.disagreement
    selectors["V1"] = aligned.v1_prediction
    selectors["LightGBM"] = aligned.lgbm_prediction

    features_confidence_v1 = aligned.v1_score.abs().groupby(aligned.fold).rank(pct=True)
    features_confidence_lgbm = aligned.lgbm_score.abs().groupby(aligned.fold).rank(pct=True)
    selectors["amplitude_rank"] = np.where(
        features_confidence_lgbm.gt(features_confidence_v1),
        aligned.lgbm_prediction,
        aligned.v1_prediction,
    )
    for estimator, selector_name in [
        ("amplitude_logit", "lower_error_amplitude"),
        ("compact_logit", "lower_error_compact"),
    ]:
        choose_lgbm = predictions[f"p_error_lgbm_{estimator}"].lt(
            predictions[f"p_error_v1_{estimator}"]
        )
        selectors[selector_name] = np.where(
            choose_lgbm, aligned.lgbm_prediction, aligned.v1_prediction
        )
    choose_direct = predictions.p_lgbm_correct_direct_compact.fillna(0.5).ge(0.5)
    selectors["direct_compact"] = np.where(
        choose_direct, aligned.lgbm_prediction, aligned.v1_prediction
    )
    return selectors


def selector_metrics(selectors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_label, block in [("overall", selectors), *list(selectors.groupby("fold"))]:
        disagreement = block.disagreement.eq(1)
        for selector in SELECTORS:
            correct = block[selector].astype(int).eq(block.truth.astype(int))
            rows.append(
                {
                    "fold": fold_label,
                    "selector": selector,
                    "n_rows": len(block),
                    "accuracy": correct.mean(),
                    "disagreement_accuracy": correct[disagreement].mean(),
                    "disagreement_rate": disagreement.mean(),
                }
            )
    return pd.DataFrame(rows)


def abstention_table(
    meta: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    aligned = meta.loc[predictions.index]
    for base_model, prediction_column in [("v1", "v1_prediction"), ("lgbm", "lgbm_prediction")]:
        correct = aligned[prediction_column].eq(aligned.truth)
        for estimator in ["amplitude_logit", "compact_logit"]:
            probability = predictions[f"p_error_{base_model}_{estimator}"]
            order = probability.sort_values().index
            for coverage in np.arange(0.1, 1.01, 0.1):
                n_keep = int(round(coverage * len(order)))
                kept = order[:n_keep]
                rows.append(
                    {
                        "base_model": base_model,
                        "estimator": estimator,
                        "coverage": coverage,
                        "n_rows": n_keep,
                        "accuracy": correct.loc[kept].mean(),
                        "mean_predicted_error": probability.loc[kept].mean(),
                    }
                )
    return pd.DataFrame(rows)


def panel_selector_metrics(selectors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PANEL_MEMBERSHIPS.exists() or selectors.fold.nunique() < 8:
        return pd.DataFrame(), pd.DataFrame()
    memberships = pd.read_csv(PANEL_MEMBERSHIPS)
    memberships = memberships[memberships.regime.eq("matched")][["panel", "TS"]]
    date_rows = []
    for selector in SELECTORS:
        correct = selectors[selector].astype(int).eq(selectors.truth.astype(int)).astype(int)
        date = pd.DataFrame({"TS": selectors.TS, "correct": correct}).groupby("TS").agg(
            n_rows=("correct", "size"), n_correct=("correct", "sum")
        ).reset_index()
        merged = memberships.merge(date, on="TS", how="left", validate="many_to_one")
        panel = merged.groupby("panel").agg(n_rows=("n_rows", "sum"), n_correct=("n_correct", "sum"))
        panel["accuracy"] = panel.n_correct / panel.n_rows
        panel["selector"] = selector
        date_rows.append(panel.reset_index())
    panel_metrics = pd.concat(date_rows, ignore_index=True)
    wide = panel_metrics.pivot(index="panel", columns="selector", values="accuracy")
    rows = []
    for selector in SELECTORS:
        gain = wide[selector] - wide["V1"]
        rows.append(
            {
                "selector": selector,
                "mean_accuracy": wide[selector].mean(),
                "mean_gain_vs_v1": gain.mean(),
                "q025_gain_vs_v1": gain.quantile(0.025),
                "median_gain_vs_v1": gain.median(),
                "q975_gain_vs_v1": gain.quantile(0.975),
                "probability_gain_positive": gain.gt(0).mean(),
            }
        )
    return panel_metrics, pd.DataFrame(rows)


def summarize_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return coefficients
    return (
        coefficients.groupby(["target", "feature"], observed=True)
        .coefficient.agg(mean_coefficient="mean", std_coefficient="std", n_folds="size")
        .reset_index()
        .assign(
            mean_abs_coefficient=lambda d: d.mean_coefficient.abs(),
            stability_ratio=lambda d: d.mean_coefficient.abs() / d.std_coefficient.replace(0, np.nan),
        )
        .sort_values(["target", "mean_abs_coefficient"], ascending=[True, False])
    )


def run(output_dir: Path, fold_limit: int | None = None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, features, numeric_columns = load_base_data()
    feature_manifest = {
        "n_rows": len(features),
        "n_numeric_features": len(numeric_columns),
        "numeric_features": numeric_columns,
        "categorical_features": CATEGORICAL_COLUMNS,
    }
    predictions, fold_metrics, coefficients = fit_cross_fitted(
        meta, features, numeric_columns, fold_limit
    )
    error_overall = overall_error_metrics(meta, predictions)
    calibration = calibration_table(meta, predictions)
    selectors = build_selectors(meta, predictions)
    selector_summary = selector_metrics(selectors)
    abstention = abstention_table(meta, predictions)
    panel_metrics, panel_summary = panel_selector_metrics(selectors)
    coefficient_summary = summarize_coefficients(coefficients)

    fold_metrics.to_csv(output_dir / "error_fold_metrics.csv", index=False)
    error_overall.to_csv(output_dir / "error_overall_metrics.csv", index=False)
    calibration.to_csv(output_dir / "calibration_curve.csv", index=False)
    predictions.to_csv(output_dir / "error_oof_predictions.csv", index_label="ROW_ID")
    selectors.to_csv(output_dir / "selector_oof_predictions.csv", index_label="ROW_ID")
    selector_summary.to_csv(output_dir / "selector_metrics.csv", index=False)
    abstention.to_csv(output_dir / "abstention_curve.csv", index=False)
    coefficients.to_csv(output_dir / "feature_coefficients_by_fold.csv", index=False)
    coefficient_summary.to_csv(output_dir / "feature_coefficients_summary.csv", index=False)
    if not panel_metrics.empty:
        panel_metrics.to_csv(output_dir / "panel_selector_metrics.csv", index=False)
        panel_summary.to_csv(output_dir / "panel_selector_summary.csv", index=False)

    overall_selectors = selector_summary[selector_summary.fold.astype(str).eq("overall")]
    fold_selectors = selector_summary[~selector_summary.fold.astype(str).eq("overall")].copy()
    v1_by_fold = fold_selectors[fold_selectors.selector.eq("V1")].set_index("fold").accuracy
    gate_rows = []
    for selector in SELECTORS[2:]:
        values = fold_selectors[fold_selectors.selector.eq(selector)].set_index("fold").accuracy
        positive_folds = int(values.sub(v1_by_fold).gt(0).sum())
        panel_probability = (
            float(panel_summary.set_index("selector").loc[selector, "probability_gain_positive"])
            if not panel_summary.empty else np.nan
        )
        gate_rows.append(
            {
                "selector": selector,
                "positive_gain_folds": positive_folds,
                "panel_probability_gain_positive": panel_probability,
                "gate_passed": bool(positive_folds >= 7 and panel_probability >= 0.90),
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(output_dir / "selector_gate.csv", index=False)

    summary = {
        "feature_manifest": feature_manifest,
        "n_completed_folds": int(selectors.fold.nunique()),
        "error_metrics": error_overall.to_dict("records"),
        "selector_overall": overall_selectors.to_dict("records"),
        "selector_gate": gate.to_dict("records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print("\nERROR PROBABILITIES")
    print(error_overall.to_string(index=False))
    print("\nSELECTORS")
    print(overall_selectors.to_string(index=False))
    print("\nGATE")
    print(gate.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold-limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_dir, fold_limit=args.fold_limit)


"""Analyses reproductibles H1, H2, H3 et incertitudes par date."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.adversarial_stress_validation import date_profile  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


OOF_FILES = {
    "Ridge V1": "v1_recheck/oof_predictions.csv",
    "Logistique V2": "classification_full/oof_logistic_C_0.003.csv",
    "LightGBM linéaire": "v3_lgbm_linear_full8/oof_lgbm_linear_returns_compact_full.csv",
    "Spécialiste volume": "volume_specialist_full8/oof_observed_returns_C_0.003.csv",
    "Forme quantile SV": "quantile_shape_full8/oof_sv_positive_shape_C_0.003.csv",
}


def load_oof(outputs_root: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(outputs_root / path, index_col="ROW_ID")
        for name, path in OOF_FILES.items()
    }


def cluster_bootstrap_models(
    frames: dict[str, pd.DataFrame],
    dates: pd.Index,
    reference: str = "Logistique V2",
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """IC cluster bootstrap par TS pour accuracy et gain apparie vs V2."""
    date_index = pd.Index(dates)
    counts = None
    correct_by_model = {}
    for name, frame in frames.items():
        local = frame[frame["TS"].isin(date_index)].copy()
        truth_column = (
            "y_true_binarized" if "y_true_binarized" in local else "target"
        )
        local["correct"] = local["prediction"].astype(int).eq(
            local[truth_column].astype(int)
        )
        grouped = local.groupby("TS", observed=True)["correct"].agg(["sum", "size"])
        grouped = grouped.reindex(date_index)
        correct_by_model[name] = grouped["sum"].to_numpy(float)
        if counts is None:
            counts = grouped["size"].to_numpy(float)

    rng = np.random.default_rng(2026)
    model_names = list(frames)
    boot_accuracy = {name: np.empty(n_bootstrap) for name in model_names}
    for iteration in range(n_bootstrap):
        sample = rng.integers(0, len(date_index), len(date_index))
        denominator = counts[sample].sum()
        for name in model_names:
            boot_accuracy[name][iteration] = (
                correct_by_model[name][sample].sum() / denominator
            )

    rows = []
    reference_boot = boot_accuracy[reference]
    for name in model_names:
        accuracy = correct_by_model[name].sum() / counts.sum()
        gain = accuracy - correct_by_model[reference].sum() / counts.sum()
        gain_boot = boot_accuracy[name] - reference_boot
        rows.append(
            {
                "model": name,
                "n_dates": len(date_index),
                "n_rows": int(counts.sum()),
                "accuracy": accuracy,
                "accuracy_ci_low": np.quantile(boot_accuracy[name], 0.025),
                "accuracy_ci_high": np.quantile(boot_accuracy[name], 0.975),
                "gain_vs_v2": gain,
                "gain_ci_low": np.quantile(gain_boot, 0.025),
                "gain_ci_high": np.quantile(gain_boot, 0.975),
            }
        )
    return pd.DataFrame(rows)


def h1_controlled_comparison(outputs_root: Path) -> pd.DataFrame:
    """Compare all-dates et complete-only sur les memes predictions OOF."""
    candidates = {
        "Toutes dates, C=0.003": outputs_root
        / "test_like_regime_cv"
        / "oof_all_dates_C_0.003.csv",
        "Dates complètes, C=0.001": outputs_root
        / "test_like_regime_cv"
        / "oof_complete_dates_only_C_0.001.csv",
        "Dates complètes, C=0.003": outputs_root
        / "test_like_regime_cv"
        / "oof_complete_dates_only_C_0.003.csv",
        "Dates complètes, C=0.01": outputs_root
        / "test_like_regime_cv"
        / "oof_complete_dates_only_C_0.01.csv",
        "Dates complètes, C=0.03": outputs_root
        / "test_like_regime_cv_weak_reg"
        / "oof_complete_dates_only_C_0.03.csv",
        "Dates complètes, C=0.1": outputs_root
        / "test_like_regime_cv_weak_reg"
        / "oof_complete_dates_only_C_0.1.csv",
    }
    frames = {name: pd.read_csv(path, index_col="ROW_ID") for name, path in candidates.items()}
    dates = pd.Index(next(iter(frames.values()))["TS"].unique())
    reference = "Toutes dates, C=0.003"
    return cluster_bootstrap_models(frames, dates, reference=reference)


def optimal_date_threshold(frame: pd.DataFrame) -> pd.Series:
    """Seuil oracle qui maximise l'accuracy sur une date (borne non deployable)."""
    ordered = frame.sort_values("score", ascending=False)
    truth = ordered["y_true_binarized"].to_numpy(dtype=int)
    scores = ordered["score"].to_numpy(float)
    cumulative_positive = np.concatenate([[0], np.cumsum(truth)])
    k = np.arange(len(truth) + 1)
    correct = cumulative_positive + ((len(truth) - k) - (truth.sum() - cumulative_positive))
    best_k = int(np.argmax(correct))
    if best_k == 0:
        threshold = float(scores[0] + 1e-12)
    elif best_k == len(scores):
        threshold = float(scores[-1] - 1e-12)
    else:
        threshold = float(0.5 * (scores[best_k - 1] + scores[best_k]))
    natural_prediction = (scores > 0.5).astype(int)
    result = {
        "n_rows": len(truth),
        "target_positive_rate": truth.mean(),
        "score_mean": scores.mean(),
        "score_std": scores.std(),
        "score_q10": np.quantile(scores, 0.10),
        "score_q25": np.quantile(scores, 0.25),
        "score_q50": np.quantile(scores, 0.50),
        "score_q75": np.quantile(scores, 0.75),
        "score_q90": np.quantile(scores, 0.90),
        "natural_positive_rate": natural_prediction.mean(),
        "natural_accuracy": (natural_prediction == truth).mean(),
        "oracle_threshold": threshold,
        "oracle_positive_rate": best_k / len(truth),
        "oracle_accuracy": correct[best_k] / len(truth),
        "auc": roc_auc_score(truth, scores) if len(np.unique(truth)) == 2 else np.nan,
    }
    return pd.Series(result)


def threshold_curve(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    thresholds = np.linspace(0.47, 0.54, 281)
    truth = frame["y_true_binarized"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(float)
    return pd.DataFrame(
        {
            "scope": scope,
            "threshold": thresholds,
            "accuracy": [((scores > value).astype(int) == truth).mean() for value in thresholds],
            "positive_rate": [(scores > value).mean() for value in thresholds],
        }
    )


def add_score_profile(profile: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    score_profile = oof.groupby("TS", observed=True)["score"].agg(
        score_mean="mean",
        score_std="std",
        score_q10=lambda values: values.quantile(0.10),
        score_q25=lambda values: values.quantile(0.25),
        score_q50=lambda values: values.quantile(0.50),
        score_q75=lambda values: values.quantile(0.75),
        score_q90=lambda values: values.quantile(0.90),
        score_positive_rate=lambda values: (values > 0.5).mean(),
    )
    return profile.join(score_profile, how="inner")


def cross_fitted_threshold_learning(
    date_features: pd.DataFrame,
    oracle: pd.DataFrame,
    oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apprend l'offset du seuil avec des features market, hors date evaluee."""
    dataset = date_features.join(oracle[["oracle_threshold"]], how="inner")
    feature_columns = list(date_features.columns)
    target = dataset["oracle_threshold"] - dataset["score_mean"]
    models = {
        "Seuil 0.5": None,
        "Offset constant cross-fitté": DummyRegressor(strategy="median"),
        "Offset Ridge market": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=100.0),
        ),
        "Offset RandomForest market": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=50,
                max_features=0.7,
                n_jobs=-1,
                random_state=0,
            ),
        ),
    }
    splitter = KFold(n_splits=8, shuffle=True, random_state=0)
    predicted_thresholds = {
        name: pd.Series(index=dataset.index, dtype=float) for name in models
    }
    predicted_thresholds["Seuil 0.5"][:] = 0.5
    for train_index, valid_index in splitter.split(dataset):
        train_dates = dataset.index[train_index]
        valid_dates = dataset.index[valid_index]
        for name, model in models.items():
            if model is None:
                continue
            model.fit(dataset.loc[train_dates, feature_columns], target.loc[train_dates])
            offset = model.predict(dataset.loc[valid_dates, feature_columns])
            predicted_thresholds[name].loc[valid_dates] = np.clip(
                dataset.loc[valid_dates, "score_mean"].to_numpy() + offset,
                0.45,
                0.56,
            )

    row_results = []
    summaries = []
    for name, thresholds in predicted_thresholds.items():
        local = oof[oof["TS"].isin(dataset.index)].copy()
        local["threshold"] = local["TS"].map(thresholds)
        local["prediction_dynamic"] = (local["score"] > local["threshold"]).astype(int)
        local["is_correct_dynamic"] = local["prediction_dynamic"].eq(
            local["y_true_binarized"].astype(int)
        )
        local["method"] = name
        row_results.append(local[["TS", "method", "threshold", "prediction_dynamic", "is_correct_dynamic"]])
        complete = local[local["n_rows"] == 276]
        summaries.append(
            {
                "method": name,
                "accuracy": local["is_correct_dynamic"].mean(),
                "accuracy_complete_dates": complete["is_correct_dynamic"].mean(),
                "prediction_positive_rate": local["prediction_dynamic"].mean(),
                "prediction_positive_rate_complete": complete[
                    "prediction_dynamic"
                ].mean(),
                "threshold_mean": local.groupby("TS")["threshold"].first().mean(),
            }
        )
    return pd.DataFrame(summaries), pd.concat(row_results, ignore_index=True)


def build_h3_features(X: pd.DataFrame) -> pd.DataFrame:
    """Profils market par date, sans y ni predictions des modeles."""
    returns = [f"RET_{lag}" for lag in range(1, 21)]
    volumes = [f"SIGNED_VOLUME_{lag}" for lag in range(1, 21)]
    grouped = X.groupby("TS", observed=True)
    parts = []
    for columns, prefix in [(returns, "ret"), (volumes, "sv")]:
        mean = grouped[columns].mean().add_prefix(f"{prefix}_cross_mean_")
        std = grouped[columns].std().add_prefix(f"{prefix}_cross_std_")
        positive = (
            X[["TS"] + columns]
            .assign(**{column: X[column].gt(0).astype(float) for column in columns})
            .groupby("TS", observed=True)[columns]
            .mean()
            .add_prefix(f"{prefix}_cross_positive_")
        )
        parts.extend([mean, std, positive])
    volume_missing = (
        X[["TS"] + volumes]
        .assign(**{column: X[column].isna().astype(float) for column in volumes})
        .groupby("TS", observed=True)[volumes]
        .mean()
        .add_prefix("sv_cross_missing_")
    )
    turnover = grouped["MEDIAN_DAILY_TURNOVER"].agg(["mean", "std", "median"])
    turnover.columns = [f"turnover_{column}" for column in turnover.columns]
    group_share = pd.crosstab(X["TS"], X["GROUP"], normalize="index")
    group_share.columns = [f"group_share_{value}" for value in group_share.columns]
    return pd.concat(parts + [volume_missing, turnover, group_share], axis=1).astype("float32")


def h3_hard_easy_classifier(
    X: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    complete_dates: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Classifie dates ou tous les modeles sont hard vs tous easy."""
    accuracies = {}
    for name, frame in frames.items():
        local = frame[frame["TS"].isin(complete_dates)].copy()
        correct = local["prediction"].astype(int).eq(local["y_true_binarized"].astype(int))
        accuracies[name] = correct.groupby(local["TS"]).mean()
    accuracy_table = pd.DataFrame(accuracies)
    hard = accuracy_table.lt(0.5).all(axis=1)
    easy = accuracy_table.gt(0.5).all(axis=1)
    labels = pd.Series(np.nan, index=accuracy_table.index)
    labels.loc[hard] = 1
    labels.loc[easy] = 0
    labels = labels.dropna().astype(int)

    features = build_h3_features(X).reindex(labels.index)
    splitter = StratifiedKFold(n_splits=8, shuffle=True, random_state=0)
    probability = pd.Series(index=labels.index, dtype=float)
    importances = []
    for train_index, valid_index in splitter.split(features, labels):
        y_train = labels.iloc[train_index]
        counts = y_train.value_counts()
        weights = y_train.map(
            {label: len(y_train) / (2 * count) for label, count in counts.items()}
        )
        model = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.03,
            n_estimators=200,
            max_depth=3,
            num_leaves=7,
            min_child_samples=50,
            min_child_weight=50.0,
            reg_alpha=5.0,
            reg_lambda=100.0,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            verbosity=-1,
            n_jobs=-1,
            random_state=0,
        )
        model.fit(
            features.iloc[train_index],
            y_train,
            sample_weight=weights,
        )
        probability.iloc[valid_index] = model.predict_proba(features.iloc[valid_index])[:, 1]
        importances.append(model.booster_.feature_importance("gain"))

    prediction = probability > 0.5
    summary = {
        "n_hard": int(labels.sum()),
        "n_easy": int((1 - labels).sum()),
        "auc": float(roc_auc_score(labels, probability)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }
    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "gain_importance": np.mean(importances, axis=0),
        }
    ).sort_values("gain_importance", ascending=False)

    characteristics = []
    for column in features:
        hard_values = features.loc[labels.eq(1), column]
        easy_values = features.loc[labels.eq(0), column]
        pooled_std = np.sqrt(0.5 * (hard_values.var() + easy_values.var()))
        standardized_difference = (
            (hard_values.mean() - easy_values.mean()) / pooled_std
            if pooled_std and np.isfinite(pooled_std)
            else 0.0
        )
        characteristics.append(
            {
                "feature": column,
                "hard_mean": hard_values.mean(),
                "easy_mean": easy_values.mean(),
                "standardized_difference": standardized_difference,
            }
        )
    characteristics_table = pd.DataFrame(characteristics).assign(
        abs_difference=lambda table: table["standardized_difference"].abs()
    ).sort_values("abs_difference", ascending=False)
    predictions = pd.DataFrame(
        {
            "TS": labels.index,
            "hard": labels.to_numpy(),
            "probability_hard": probability.to_numpy(),
            "prediction_hard": prediction.astype(int).to_numpy(),
            "mean_model_accuracy": accuracy_table.loc[labels.index].mean(axis=1).to_numpy(),
        }
    )
    return importance, characteristics_table, predictions, summary


def run(outputs_root: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    frames = load_oof(outputs_root)
    all_dates = pd.Index(X["TS"].unique())
    date_sizes = X.groupby("TS", observed=True).size()
    complete_dates = date_sizes[date_sizes.eq(276)].index

    global_uncertainty = cluster_bootstrap_models(frames, all_dates)
    global_uncertainty["scope"] = "Toutes dates"
    complete_uncertainty = cluster_bootstrap_models(frames, complete_dates)
    complete_uncertainty["scope"] = "Dates complètes (276)"
    uncertainty = pd.concat([global_uncertainty, complete_uncertainty], ignore_index=True)
    uncertainty.to_csv(output_dir / "model_uncertainty.csv", index=False)

    h1 = h1_controlled_comparison(outputs_root)
    h1.to_csv(output_dir / "h1_training_regime.csv", index=False)

    v2 = frames["Logistique V2"].copy()
    v2["n_rows"] = v2["TS"].map(date_sizes)
    auc_by_fold = (
        v2.groupby("fold", observed=True)
        .apply(lambda frame: roc_auc_score(frame["y_true_binarized"], frame["score"]), include_groups=False)
        .rename("auc")
        .reset_index()
    )
    oracle = v2.groupby("TS", observed=True).apply(optimal_date_threshold, include_groups=False)
    oracle.to_csv(output_dir / "h2_oracle_threshold_by_date.csv")
    auc_by_fold.to_csv(output_dir / "h2_auc_by_fold.csv", index=False)
    curves = pd.concat(
        [
            threshold_curve(v2, "Toutes dates"),
            threshold_curve(v2[v2["TS"].isin(complete_dates)], "Dates complètes"),
        ],
        ignore_index=True,
    )
    curves.to_csv(output_dir / "h2_threshold_curve.csv", index=False)

    market_profile = add_score_profile(date_profile(X), v2)
    threshold_summary, threshold_rows = cross_fitted_threshold_learning(
        market_profile,
        oracle,
        v2,
    )
    threshold_summary.to_csv(output_dir / "h2_learned_thresholds.csv", index=False)
    threshold_rows.to_csv(output_dir / "h2_learned_threshold_rows.csv", index=False)

    importance, characteristics, h3_predictions, h3_summary = h3_hard_easy_classifier(
        X, frames, complete_dates
    )
    importance.to_csv(output_dir / "h3_feature_importance.csv", index=False)
    characteristics.to_csv(output_dir / "h3_characteristics.csv", index=False)
    h3_predictions.to_csv(output_dir / "h3_oof_predictions.csv", index=False)

    summary = {
        "H1": {
            "all_dates_accuracy": float(
                h1.loc[h1["model"].eq("Toutes dates, C=0.003"), "accuracy"].iloc[0]
            ),
            "best_complete_only": h1[
                h1["model"].str.startswith("Dates complètes")
            ].sort_values("accuracy", ascending=False)
            .iloc[0]
            .to_dict(),
        },
        "H2": {
            "global_auc": float(roc_auc_score(v2["y_true_binarized"], v2["score"])),
            "mean_fold_auc": float(auc_by_fold["auc"].mean()),
            "natural_accuracy": float(v2["is_correct"].mean()),
            "oracle_date_accuracy": float(
                np.average(oracle["oracle_accuracy"], weights=oracle["n_rows"])
            ),
            "natural_date_accuracy_recomputed": float(
                np.average(oracle["natural_accuracy"], weights=oracle["n_rows"])
            ),
            "learned_thresholds": threshold_summary.to_dict("records"),
        },
        "H3": h3_summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "hypotheses_h1_h2_h3",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.outputs_root, args.output_dir)

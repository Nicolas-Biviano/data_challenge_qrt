"""Analyse détaillée X-only des lignes où cinq modèles OOF ont tous tort/raison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.hypotheses_h1_h2_h3 import OOF_FILES, load_oof  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


QUANTILES = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])


def row_features(X: pd.DataFrame) -> pd.DataFrame:
    """Features de ligne brutes et résumés historiques/transversaux, tous X-only."""
    returns = [f"RET_{lag}" for lag in range(1, 21)]
    volumes = [f"SIGNED_VOLUME_{lag}" for lag in range(1, 21)]
    result = X[returns + volumes + ["MEDIAN_DAILY_TURNOVER"]].astype("float32").copy()

    recent_returns = X[[f"RET_{lag}" for lag in range(1, 6)]]
    old_returns = X[[f"RET_{lag}" for lag in range(6, 21)]]
    result["ret_recent_mean_1_5"] = recent_returns.mean(axis=1)
    result["ret_old_mean_6_20"] = old_returns.mean(axis=1)
    result["ret_recent_std_1_5"] = recent_returns.std(axis=1)
    result["ret_old_std_6_20"] = old_returns.std(axis=1)
    result["ret_mean_gap_recent_old"] = result.ret_recent_mean_1_5 - result.ret_old_mean_6_20
    result["ret_positive_share_20"] = X[returns].gt(0).mean(axis=1)
    result["ret_abs_mean_20"] = X[returns].abs().mean(axis=1)
    result["ret_range_20"] = X[returns].max(axis=1) - X[returns].min(axis=1)
    return_signs = np.sign(X[returns].to_numpy(float))
    result["ret_sign_changes_20"] = np.sum(return_signs[:, 1:] != return_signs[:, :-1], axis=1)

    result["sv_observed_count_20"] = X[volumes].notna().sum(axis=1)
    result["sv_recent_observed_count_5"] = X[[f"SIGNED_VOLUME_{lag}" for lag in range(1, 6)]].notna().sum(axis=1)
    result["sv_mean_observed_20"] = X[volumes].mean(axis=1)
    result["sv_std_observed_20"] = X[volumes].std(axis=1)
    result["sv_abs_mean_observed_20"] = X[volumes].abs().mean(axis=1)
    result["sv_positive_share_observed_20"] = X[volumes].gt(0).sum(axis=1) / X[volumes].notna().sum(axis=1).clip(lower=1)
    for column in volumes:
        result[f"missing_{column}"] = X[column].isna().astype("float32")

    turnover = X["MEDIAN_DAILY_TURNOVER"].clip(lower=0)
    result["log1p_turnover"] = np.log1p(turnover)
    result["date_n_rows"] = X["TS"].map(X.groupby("TS", observed=True).size()).astype("float32")

    result["xsec_rank_RET_1"] = X.groupby("TS", observed=True)["RET_1"].rank(pct=True)
    result["xsec_rank_ret_recent_mean"] = result.groupby(X["TS"], observed=True)["ret_recent_mean_1_5"].rank(pct=True)
    result["xsec_rank_turnover"] = X.groupby("TS", observed=True)["MEDIAN_DAILY_TURNOVER"].rank(pct=True)
    result["xsec_rank_SIGNED_VOLUME_1"] = X.groupby("TS", observed=True)["SIGNED_VOLUME_1"].rank(pct=True)
    result["group_date_rank_RET_1"] = X.groupby(["TS", "GROUP"], observed=True)["RET_1"].rank(pct=True)
    result["group_date_rank_turnover"] = X.groupby(["TS", "GROUP"], observed=True)["MEDIAN_DAILY_TURNOVER"].rank(pct=True)
    return result.astype("float32")


def feature_family(feature: str) -> str:
    lower = feature.lower()
    if feature.startswith("RET_") or feature.startswith("ret_"):
        return "returns"
    if feature.startswith("SIGNED_VOLUME_") or feature.startswith("sv_") or feature.startswith("missing_SIGNED_VOLUME_"):
        return "signed_volume"
    if feature.startswith("xsec_") or feature.startswith("group_date_"):
        return "structure_transversale"
    if "turnover" in lower:
        return "turnover"
    if feature == "date_n_rows":
        return "taille_date"
    return "autre"


def psi_from_quantile_bins(a: np.ndarray, b: np.ndarray) -> float:
    combined = np.concatenate([a, b])
    edges = np.unique(np.quantile(combined, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, bins=edges)[0].astype(float)
    pb = np.histogram(b, bins=edges)[0].astype(float)
    pa = (pa + 0.5) / (pa.sum() + 0.5 * len(pa))
    pb = (pb + 0.5) / (pb.sum() + 0.5 * len(pb))
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def distribution_metrics(
    features: pd.DataFrame,
    labels: pd.Series,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Effets de distribution globaux puis conditionnels à chaque classe cible."""
    metric_rows = []
    quantile_rows = []
    scopes = [("Tous", pd.Series(True, index=labels.index)), ("y=0", target.eq(0)), ("y=1", target.eq(1))]
    for scope, scope_mask in scopes:
        hard_mask = labels.eq(1) & scope_mask
        easy_mask = labels.eq(0) & scope_mask
        for column in features.columns:
            hard_raw = features.loc[hard_mask, column].to_numpy(float)
            easy_raw = features.loc[easy_mask, column].to_numpy(float)
            hard = hard_raw[np.isfinite(hard_raw)]
            easy = easy_raw[np.isfinite(easy_raw)]
            missing_hard = 1 - len(hard) / len(hard_raw)
            missing_easy = 1 - len(easy) / len(easy_raw)
            if len(hard) == 0 or len(easy) == 0:
                continue
            q_hard = np.quantile(hard, QUANTILES)
            q_easy = np.quantile(easy, QUANTILES)
            pooled_std = np.sqrt(0.5 * (np.var(hard, ddof=1) + np.var(easy, ddof=1)))
            pooled_robust_scale = 0.5 * ((q_hard[5] - q_hard[3]) + (q_easy[5] - q_easy[3])) / 1.349
            combined_iqr = np.quantile(np.concatenate([hard, easy]), 0.75) - np.quantile(np.concatenate([hard, easy]), 0.25)
            smd = (hard.mean() - easy.mean()) / pooled_std if pooled_std > 0 else 0.0
            robust_smd = (q_hard[4] - q_easy[4]) / pooled_robust_scale if pooled_robust_scale > 0 else 0.0
            quantile_wasserstein = np.mean(np.abs(q_hard - q_easy)) / combined_iqr if combined_iqr > 0 else 0.0
            ks = ks_2samp(hard, easy, method="asymp").statistic
            metric_rows.append(
                {
                    "scope": scope,
                    "feature": column,
                    "family": feature_family(column),
                    "n_all_wrong": len(hard_raw),
                    "n_all_right": len(easy_raw),
                    "mean_all_wrong": hard.mean(),
                    "mean_all_right": easy.mean(),
                    "median_all_wrong": q_hard[4],
                    "median_all_right": q_easy[4],
                    "smd": smd,
                    "robust_smd": robust_smd,
                    "ks": ks,
                    "quantile_wasserstein_iqr": quantile_wasserstein,
                    "psi": psi_from_quantile_bins(hard, easy),
                    "missing_all_wrong": missing_hard,
                    "missing_all_right": missing_easy,
                    "missing_diff_pp": 100 * (missing_hard - missing_easy),
                }
            )
            for group_name, values in [("Tous faux", q_hard), ("Tous justes", q_easy)]:
                for quantile, value in zip(QUANTILES, values):
                    quantile_rows.append(
                        {"scope": scope, "feature": column, "group": group_name, "quantile": quantile, "value": value}
                    )
    metrics = pd.DataFrame(metric_rows)
    metrics["separation_score"] = metrics[["smd", "robust_smd"]].abs().max(axis=1).clip(upper=5)
    metrics["separation_score"] += metrics["ks"] + metrics["missing_diff_pp"].abs() / 100
    return metrics.sort_values(["scope", "separation_score"], ascending=[True, False]), pd.DataFrame(quantile_rows)


def cluster_bootstrap_smd(
    X: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    top_features: list[str],
    n_bootstrap: int = 800,
) -> pd.DataFrame:
    """IC 95 % du SMD, en rééchantillonnant les dates entières."""
    dates = pd.Index(X["TS"].unique())
    rng = np.random.default_rng(2026)
    bootstrap_indices = rng.integers(0, len(dates), size=(n_bootstrap, len(dates)))
    rows = []
    for feature in top_features:
        local = pd.DataFrame({"TS": X["TS"], "label": labels, "value": features[feature]})
        local = local[np.isfinite(local.value)]
        local["square"] = local.value.astype(float) ** 2
        agg = local.groupby(["TS", "label"], observed=True).agg(
            total=("value", "sum"), square=("square", "sum"), n=("value", "size")
        )
        arrays = {}
        for label in [0, 1]:
            block = agg.xs(label, level="label", drop_level=True).reindex(dates).fillna(0)
            arrays[label] = block[["total", "square", "n"]].to_numpy(float)
        boot = np.empty(n_bootstrap)
        for iteration, sample in enumerate(bootstrap_indices):
            stats = {}
            for label in [0, 1]:
                total, square, count = arrays[label][sample].sum(axis=0)
                mean = total / count
                variance = max(square / count - mean**2, 0.0)
                stats[label] = (mean, variance)
            pooled = np.sqrt(0.5 * (stats[0][1] + stats[1][1]))
            boot[iteration] = (stats[1][0] - stats[0][0]) / pooled if pooled > 0 else 0.0
        observed = distribution_smd(features.loc[labels.eq(1), feature], features.loc[labels.eq(0), feature])
        rows.append(
            {"feature": feature, "smd": observed, "smd_ci_low": np.quantile(boot, .025), "smd_ci_high": np.quantile(boot, .975)}
        )
    return pd.DataFrame(rows)


def distribution_smd(hard: pd.Series, easy: pd.Series) -> float:
    hard = hard.dropna().to_numpy(float)
    easy = easy.dropna().to_numpy(float)
    pooled = np.sqrt(0.5 * (np.var(hard, ddof=1) + np.var(easy, ddof=1)))
    return float((hard.mean() - easy.mean()) / pooled) if pooled > 0 else 0.0


def categorical_metrics(
    X: pd.DataFrame,
    labels: pd.Series,
    target: pd.Series,
    fold: pd.Series,
) -> pd.DataFrame:
    rows = []
    scopes = [("Tous", pd.Series(True, index=labels.index)), ("y=0", target.eq(0)), ("y=1", target.eq(1))]
    for column in ["GROUP", "ALLOCATION"]:
        for scope, mask in scopes:
            local = pd.DataFrame({"category": X[column].astype(str), "label": labels, "fold": fold})[mask]
            global_rate = local.label.mean()
            counts = pd.crosstab(local.category, local.label).reindex(columns=[0, 1], fill_value=0)
            fold_rates = local.groupby(["category", "fold"], observed=True).label.mean().unstack()
            fold_global = local.groupby("fold", observed=True).label.mean()
            for category, count in counts.iterrows():
                n_right, n_wrong = int(count[0]), int(count[1])
                n_total = n_right + n_wrong
                rates = fold_rates.loc[category].dropna()
                comparable = rates.index.intersection(fold_global.index)
                rows.append(
                    {
                        "scope": scope,
                        "column": column,
                        "category": category,
                        "n_extreme": n_total,
                        "n_all_wrong": n_wrong,
                        "n_all_right": n_right,
                        "all_wrong_rate": n_wrong / n_total,
                        "risk_diff_pp": 100 * (n_wrong / n_total - global_rate),
                        "fold_rate_std": rates.std(ddof=1),
                        "fold_rate_min": rates.min(),
                        "fold_rate_max": rates.max(),
                        "direction_consistency": np.mean(rates.loc[comparable] > fold_global.loc[comparable]),
                    }
                )
    return pd.DataFrame(rows)


def model_score_diagnostics(v2: pd.DataFrame, labels: pd.Series, target: pd.Series) -> pd.DataFrame:
    local = pd.DataFrame({"label": labels, "target": target, "score": v2["score"].astype(float)})
    local["margin_abs"] = (local.score - .5).abs()
    rows = []
    for label, label_name in [(0, "Tous justes"), (1, "Tous faux")]:
        for target_value in [0, 1]:
            block = local[local.label.eq(label) & local.target.eq(target_value)]
            rows.append(
                {
                    "group": label_name,
                    "target": target_value,
                    "n": len(block),
                    "score_mean": block.score.mean(),
                    "score_q10": block.score.quantile(.10),
                    "score_median": block.score.median(),
                    "score_q90": block.score.quantile(.90),
                    "margin_abs_mean": block.margin_abs.mean(),
                    "within_001_of_threshold": block.margin_abs.le(.01).mean(),
                    "within_002_of_threshold": block.margin_abs.le(.02).mean(),
                }
            )
    return pd.DataFrame(rows)


def error_gradient(
    X: pd.DataFrame,
    target: pd.Series,
    error_count: pd.Series,
    v2: pd.DataFrame,
) -> pd.DataFrame:
    """Profil des lignes lorsque le nombre de modèles en erreur augmente."""
    sign = 2 * target - 1
    local = pd.DataFrame(
        {
            "error_count": error_count,
            "target": target,
            "aligned_RET_1": sign * X["RET_1"],
            "abs_RET_1": X["RET_1"].abs(),
            "turnover": X["MEDIAN_DAILY_TURNOVER"],
            "sv1_missing": X["SIGNED_VOLUME_1"].isna().astype(float),
            "v2_score": v2["score"].astype(float),
        }
    )
    local["v2_aligned_margin"] = sign * (local.v2_score - .5)
    return (
        local.groupby("error_count", observed=True)
        .agg(
            n=("target", "size"),
            target_positive_rate=("target", "mean"),
            aligned_RET_1_mean=("aligned_RET_1", "mean"),
            aligned_RET_1_median=("aligned_RET_1", "median"),
            abs_RET_1_median=("abs_RET_1", "median"),
            turnover_median=("turnover", "median"),
            sv1_missing_rate=("sv1_missing", "mean"),
            v2_aligned_margin_mean=("v2_aligned_margin", "mean"),
            v2_score_mean=("v2_score", "mean"),
        )
        .reset_index()
    )


def category_behavior(
    X: pd.DataFrame,
    target: pd.Series,
    labels: pd.Series,
    error_count: pd.Series,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Taux de cible, de prédiction et d'erreur par GROUP/ALLOCATION."""
    base = pd.DataFrame({"target": target, "all_wrong": labels, "error_count": error_count})
    for name, frame in frames.items():
        slug = (
            name.lower()
            .replace(" ", "_")
            .replace("é", "e")
            .replace("è", "e")
            .replace("quantile_sv", "quantile")
        )
        prediction = frame["prediction"].astype(int)
        base[f"prediction_positive_{slug}"] = prediction
        base[f"accuracy_{slug}"] = prediction.eq(target).astype(float)
    rows = []
    for column in ["GROUP", "ALLOCATION"]:
        local = base.assign(category=X[column].astype(str), category_type=column)
        grouped = local.groupby(["category_type", "category"], observed=True)
        summary = grouped.agg(
            n=("target", "size"),
            n_extreme=("all_wrong", "count"),
            target_positive_rate=("target", "mean"),
            mean_error_count=("error_count", "mean"),
            extreme_share=("all_wrong", lambda values: values.notna().mean()),
            all_wrong_rate_extreme=("all_wrong", "mean"),
        )
        for name in frames:
            slug = (
                name.lower()
                .replace(" ", "_")
                .replace("é", "e")
                .replace("è", "e")
                .replace("quantile_sv", "quantile")
            )
            summary[f"prediction_positive_{slug}"] = grouped[f"prediction_positive_{slug}"].mean()
            summary[f"accuracy_{slug}"] = grouped[f"accuracy_{slug}"].mean()
        rows.append(summary.reset_index())
    return pd.concat(rows, ignore_index=True)


def row_classifier(
    X: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test X-only hors échantillon, avec dates entières dans chaque fold."""
    categorical = pd.DataFrame(index=X.index)
    categorical["ALLOCATION"] = X["ALLOCATION"].astype("category")
    categorical["GROUP"] = X["GROUP"].astype(str).astype("category")
    all_features = pd.concat([features, categorical], axis=1)
    return_columns = [column for column in features if feature_family(column) == "returns"]
    volume_columns = [column for column in features if feature_family(column) == "signed_volume"]
    structure_columns = [column for column in features if feature_family(column) in {"structure_transversale", "turnover", "taille_date"}]
    blocks = {
        "Returns": return_columns,
        "Volumes": volume_columns,
        "Structure + catégories": structure_columns + ["ALLOCATION", "GROUP"],
        "Toutes X enrichies": list(all_features.columns),
    }
    extreme = labels.notna()
    y = labels[extreme].astype(int)
    groups = X.loc[extreme, "TS"]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    predictions = []
    importances = []
    for block_name, columns in blocks.items():
        probability = pd.Series(index=y.index, dtype=float)
        fold_number = pd.Series(index=y.index, dtype=int)
        for fold_index, (train_index, valid_index) in enumerate(splitter.split(all_features.loc[extreme, columns], y, groups), start=1):
            train_rows, valid_rows = y.index[train_index], y.index[valid_index]
            y_train = y.loc[train_rows]
            counts = y_train.value_counts()
            weights = y_train.map({value: len(y_train) / (2 * count) for value, count in counts.items()})
            model = lgb.LGBMClassifier(
                objective="binary",
                learning_rate=0.03,
                n_estimators=220,
                max_depth=4,
                num_leaves=12,
                min_child_samples=500,
                min_child_weight=100.0,
                reg_alpha=10.0,
                reg_lambda=150.0,
                min_split_gain=0.01,
                max_bin=63,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.7,
                extra_trees=True,
                verbosity=-1,
                n_jobs=-1,
                random_state=fold_index,
            )
            model.fit(all_features.loc[train_rows, columns], y_train, sample_weight=weights)
            probability.loc[valid_rows] = model.predict_proba(all_features.loc[valid_rows, columns])[:, 1]
            fold_number.loc[valid_rows] = fold_index
            gains = model.booster_.feature_importance("gain")
            importances.extend(
                {"block": block_name, "fold": fold_index, "feature": feature, "gain_importance": gain}
                for feature, gain in zip(columns, gains)
            )
        predictions.append(
            pd.DataFrame(
                {
                    "ROW_ID": y.index,
                    "TS": groups,
                    "target": target.loc[y.index],
                    "all_wrong": y,
                    "block": block_name,
                    "fold": fold_number,
                    "probability_all_wrong": probability,
                }
            )
        )
    prediction_table = pd.concat(predictions, ignore_index=True)
    metric_rows = []
    for block_name, block in prediction_table.groupby("block", sort=False):
        prediction = block.probability_all_wrong.gt(.5)
        metric_rows.append(
            {
                "block": block_name,
                "scope": "Tous",
                "n": len(block),
                "prevalence_all_wrong": block.all_wrong.mean(),
                "auc": roc_auc_score(block.all_wrong, block.probability_all_wrong),
                "average_precision": average_precision_score(block.all_wrong, block.probability_all_wrong),
                "accuracy": accuracy_score(block.all_wrong, prediction),
                "balanced_accuracy": balanced_accuracy_score(block.all_wrong, prediction),
            }
        )
        for target_value in [0, 1]:
            subset = block[block.target.eq(target_value)]
            prediction_subset = subset.probability_all_wrong.gt(.5)
            metric_rows.append(
                {
                    "block": block_name,
                    "scope": f"y={target_value}",
                    "n": len(subset),
                    "prevalence_all_wrong": subset.all_wrong.mean(),
                    "auc": roc_auc_score(subset.all_wrong, subset.probability_all_wrong),
                    "average_precision": average_precision_score(subset.all_wrong, subset.probability_all_wrong),
                    "accuracy": accuracy_score(subset.all_wrong, prediction_subset),
                    "balanced_accuracy": balanced_accuracy_score(subset.all_wrong, prediction_subset),
                }
            )
    importance_table = (
        pd.DataFrame(importances)
        .groupby(["block", "feature"], as_index=False).gain_importance.mean()
        .sort_values(["block", "gain_importance"], ascending=[True, False])
    )
    return pd.DataFrame(metric_rows), prediction_table, importance_table


def run(outputs_root: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()["target_binarized"].astype(int)
    frames = load_oof(outputs_root)

    correctness = []
    for name, frame in frames.items():
        if not frame.index.equals(X.index):
            raise ValueError(f"Index OOF non aligné pour {name}")
        if not frame["y_true_binarized"].astype(int).equals(y):
            raise ValueError(f"Cible OOF non alignée pour {name}")
        correctness.append(frame["prediction"].astype(int).eq(y).rename(name))
    correctness = pd.concat(correctness, axis=1)
    error_count = (~correctness).sum(axis=1).astype(int)
    labels = pd.Series(np.nan, index=X.index, name="all_wrong")
    labels.loc[error_count.eq(0)] = 0
    labels.loc[error_count.eq(len(frames))] = 1

    consensus = pd.DataFrame({"TS": X.TS, "target": y, "error_count": error_count, "all_wrong": labels})
    consensus.to_csv(output_dir / "h3_row_consensus.csv", index_label="ROW_ID")
    error_summary = (
        consensus.groupby("error_count", observed=True)
        .agg(n=("target", "size"), target_positive_rate=("target", "mean"), n_dates=("TS", "nunique"))
        .reset_index()
    )
    error_summary["share"] = error_summary.n / len(consensus)
    error_summary.to_csv(output_dir / "h3_row_error_count_summary.csv", index=False)

    gradient = error_gradient(X, y, error_count, frames["Logistique V2"])
    gradient.to_csv(output_dir / "h3_row_error_gradient.csv", index=False)

    behavior = category_behavior(X, y, labels, error_count, frames)
    behavior.to_csv(output_dir / "h3_row_category_behavior.csv", index=False)

    features = row_features(X)
    extreme_labels = labels.dropna().astype(int)
    metrics, quantiles = distribution_metrics(features.loc[extreme_labels.index], extreme_labels, y.loc[extreme_labels.index])
    metrics.to_csv(output_dir / "h3_row_distribution_metrics.csv", index=False)
    quantiles.to_csv(output_dir / "h3_row_quantiles.csv", index=False)

    top_features = metrics[metrics.scope.eq("Tous")].head(20).feature.tolist()
    bootstrap = cluster_bootstrap_smd(X.loc[extreme_labels.index], features.loc[extreme_labels.index], extreme_labels, top_features)
    bootstrap.to_csv(output_dir / "h3_row_smd_bootstrap.csv", index=False)

    fold = frames["Logistique V2"]["fold"].astype(int)
    categories = categorical_metrics(X.loc[extreme_labels.index], extreme_labels, y.loc[extreme_labels.index], fold.loc[extreme_labels.index])
    categories.to_csv(output_dir / "h3_row_categorical_metrics.csv", index=False)

    score_diagnostics = model_score_diagnostics(frames["Logistique V2"].loc[extreme_labels.index], extreme_labels, y.loc[extreme_labels.index])
    score_diagnostics.to_csv(output_dir / "h3_row_v2_score_diagnostics.csv", index=False)

    classifier_metrics, classifier_predictions, classifier_importance = row_classifier(X, features, labels, y)
    classifier_metrics.to_csv(output_dir / "h3_row_classifier_metrics.csv", index=False)
    classifier_predictions.to_csv(output_dir / "h3_row_classifier_predictions.csv", index=False)
    classifier_importance.to_csv(output_dir / "h3_row_classifier_importance.csv", index=False)

    summary = {
        "n_rows": len(X),
        "n_all_right": int(labels.eq(0).sum()),
        "n_all_wrong": int(labels.eq(1).sum()),
        "n_mixed": int(labels.isna().sum()),
        "extreme_share": float(labels.notna().mean()),
        "target_positive_all_right": float(y[labels.eq(0)].mean()),
        "target_positive_all_wrong": float(y[labels.eq(1)].mean()),
        "top_distribution_features": metrics[metrics.scope.eq("Tous")].head(20).to_dict("records"),
        "classifier": classifier_metrics.to_dict("records"),
    }
    with (output_dir / "h3_row_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "top_distribution_features"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "h3_row_error_analysis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.outputs_root, args.output_dir)

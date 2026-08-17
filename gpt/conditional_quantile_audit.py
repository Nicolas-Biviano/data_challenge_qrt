"""Audit des moyennes conditionnelles par quantile sur les residus OOF V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.model_v2 import REFERENCE_OOF  # noqa: E402
from gpt.v3_features import build_v3_features  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


def quantile_codes(values: pd.Series, n_quantiles: int) -> pd.Series:
    """Assigne des quantiles; les valeurs manquantes restent manquantes."""
    observed = values.dropna()
    codes = pd.Series(np.nan, index=values.index, dtype="float64")
    if observed.nunique() < 2:
        return codes
    if observed.nunique() <= n_quantiles:
        ordered = {value: index for index, value in enumerate(sorted(observed.unique()))}
        codes.loc[observed.index] = observed.map(ordered).astype(float)
        return codes
    codes.loc[observed.index] = pd.qcut(
        observed,
        q=n_quantiles,
        labels=False,
        duplicates="drop",
    ).astype(float)
    return codes


def profile_feature(
    feature_name: str,
    values: pd.Series,
    audit: pd.DataFrame,
    n_quantiles: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Construit le profil d'une feature et ses diagnostics de stabilite."""
    bin_code = quantile_codes(values, n_quantiles)
    observed = bin_code.notna()
    local = audit.loc[observed].copy()
    local["feature_value"] = values.loc[observed].astype(float)
    local["quantile"] = bin_code.loc[observed].astype(int)
    profile = (
        local.groupby("quantile", observed=True)
        .agg(
            n=("residual", "size"),
            value_min=("feature_value", "min"),
            value_mean=("feature_value", "mean"),
            value_max=("feature_value", "max"),
            target_positive_rate=("y_true_binarized", "mean"),
            target_mean=("y_true", "mean"),
            baseline_probability=("score", "mean"),
            residual_mean=("residual", "mean"),
            residual_std=("residual", "std"),
            baseline_accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    profile["residual_se"] = profile["residual_std"] / np.sqrt(profile["n"])
    profile.insert(0, "feature", feature_name)

    n_bins = len(profile)
    if n_bins >= 2:
        monotonic_corr = float(
            profile["quantile"].corr(profile["residual_mean"], method="spearman")
        )
        residual_range = float(
            profile["residual_mean"].max() - profile["residual_mean"].min()
        )
        middle = profile.iloc[1:-1] if n_bins > 2 else profile
        edge_contrast = float(
            profile.iloc[[0, -1]]["residual_mean"].mean()
            - middle["residual_mean"].mean()
        )
        max_abs_z = float(
            (profile["residual_mean"].abs() / profile["residual_se"]).max()
        )
    else:
        monotonic_corr = np.nan
        residual_range = 0.0
        edge_contrast = np.nan
        max_abs_z = np.nan

    global_pattern = profile.set_index("quantile")["residual_mean"]
    fold_patterns = (
        local.groupby(["fold", "quantile"], observed=True)["residual"]
        .mean()
        .unstack("quantile")
        .reindex(columns=global_pattern.index)
    )
    correlations = fold_patterns.apply(
        lambda row: row.corr(global_pattern) if row.notna().sum() >= 3 else np.nan,
        axis=1,
    )
    signs = np.sign(fold_patterns)
    global_sign = np.sign(global_pattern)
    sign_agreement = signs.eq(global_sign, axis="columns").mean(axis=1)
    mean_fold_corr = float(correlations.mean()) if correlations.notna().any() else np.nan
    mean_sign_agreement = float(sign_agreement.mean())
    stable_amplitude = residual_range * max(0.0, mean_fold_corr if np.isfinite(mean_fold_corr) else 0.0)

    missing = values.isna()
    summary: dict[str, float | int | str] = {
        "feature": feature_name,
        "n_observed": int(observed.sum()),
        "missing_rate": float(missing.mean()),
        "n_bins": n_bins,
        "residual_range": residual_range,
        "monotonic_spearman": monotonic_corr,
        "edge_vs_middle": edge_contrast,
        "max_abs_bin_z": max_abs_z,
        "mean_fold_pattern_corr": mean_fold_corr,
        "mean_fold_sign_agreement": mean_sign_agreement,
        "stable_amplitude": stable_amplitude,
        "missing_residual_mean": float(audit.loc[missing, "residual"].mean())
        if missing.any()
        else np.nan,
        "observed_residual_mean": float(audit.loc[~missing, "residual"].mean()),
    }
    return profile, summary


def interaction_profile(
    feature: str,
    values: pd.Series,
    ret1: pd.Series,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Moyenne residuelle dans une grille quintile RET_1 x quintile feature."""
    feature_bin = quantile_codes(values, 5)
    ret1_bin = quantile_codes(ret1, 5)
    observed = feature_bin.notna() & ret1_bin.notna()
    local = audit.loc[observed].copy()
    local["feature_quantile"] = feature_bin.loc[observed].astype(int)
    local["ret1_quantile"] = ret1_bin.loc[observed].astype(int)
    result = (
        local.groupby(["ret1_quantile", "feature_quantile"], observed=True)
        .agg(
            n=("residual", "size"),
            residual_mean=("residual", "mean"),
            target_positive_rate=("y_true_binarized", "mean"),
            baseline_probability=("score", "mean"),
        )
        .reset_index()
    )
    result.insert(0, "feature", feature)
    return result


def run_audit(
    output_dir: Path,
    reference_oof: Path,
    n_quantiles: int,
    top_interactions: int,
) -> pd.DataFrame:
    """Execute l'audit et sauvegarde profils, resumes et interactions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    features, blocks = build_v3_features(X)
    oof = pd.read_csv(reference_oof, index_col="ROW_ID")
    oof = oof.reindex(features.index)
    audit = oof[[
        "fold",
        "y_true",
        "y_true_binarized",
        "score",
        "is_correct",
    ]].copy()
    audit["residual"] = audit["y_true_binarized"] - audit["score"]

    profiles = []
    summaries = []
    for feature in blocks["non_return"]:
        profile, summary = profile_feature(
            feature,
            features[feature],
            audit,
            n_quantiles,
        )
        profiles.append(profile)
        summaries.append(summary)

    summary_table = pd.DataFrame(summaries).sort_values(
        ["stable_amplitude", "residual_range"],
        ascending=False,
    )
    profile_table = pd.concat(profiles, ignore_index=True)
    eligible = summary_table.query("n_bins >= 4 and n_observed >= 10000")
    top_features = eligible.head(top_interactions)["feature"].tolist()
    interactions = [
        interaction_profile(feature, features[feature], features["RET_1"], audit)
        for feature in top_features
    ]
    interaction_table = pd.concat(interactions, ignore_index=True)

    summary_table.to_csv(output_dir / "feature_summary.csv", index=False)
    profile_table.to_csv(output_dir / "quantile_profiles.csv", index=False)
    interaction_table.to_csv(output_dir / "ret1_interactions.csv", index=False)
    with (output_dir / "audit_config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "reference_oof": str(reference_oof),
                "n_quantiles": n_quantiles,
                "top_interactions": top_features,
                "target_encoding": False,
                "date_order_used": False,
                "ranking_target": "out-of-fold classification residual",
            },
            stream,
            indent=2,
        )
    return summary_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "conditional_quantile_audit",
    )
    parser.add_argument("--reference-oof", type=Path, default=REFERENCE_OOF)
    parser.add_argument("--n-quantiles", type=int, default=10)
    parser.add_argument("--top-interactions", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    table = run_audit(
        args.output_dir,
        args.reference_oof,
        args.n_quantiles,
        args.top_interactions,
    )
    print(
        table.head(15)[
            [
                "feature",
                "missing_rate",
                "residual_range",
                "monotonic_spearman",
                "mean_fold_pattern_corr",
                "mean_fold_sign_agreement",
                "stable_amplitude",
            ]
        ].to_string(index=False)
    )

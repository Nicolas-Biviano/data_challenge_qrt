"""H9: X-only allocation archetypes and relational date profiles for H8 slopes."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.adversarial_stress_validation import date_profile  # noqa: E402
from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from gpt.h8_slope_inversion import (  # noqa: E402
    build_target_matrix,
    compute_date_slopes,
    safe_corr,
)
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h9_allocation_archetypes_relational"
H8_METRICS = GPT_DIR / "outputs" / "h8_slope_inversion" / "feature_metrics.csv"
N_ARCHETYPES = 6


def allocation_behavior_profile(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    work = X.copy()
    group = work.groupby("ALLOCATION", observed=True)
    profile = pd.DataFrame(index=pd.Index(sorted(work.ALLOCATION.unique()), name="ALLOCATION"))
    for feature in BASE_RETURNS:
        profile[f"{feature}_mean"] = group[feature].mean()
        profile[f"{feature}_std"] = group[feature].std()
        profile[f"{feature}_abs_mean"] = work[feature].abs().groupby(work.ALLOCATION).mean()
    volume_columns = [f"SIGNED_VOLUME_{lag}" for lag in range(1, 21)]
    profile["turnover_median"] = group.MEDIAN_DAILY_TURNOVER.median()
    profile["turnover_mean"] = group.MEDIAN_DAILY_TURNOVER.mean()
    profile["volume_observed_fraction"] = work[volume_columns].notna().mean(axis=1).groupby(
        work.ALLOCATION
    ).mean()
    profile["sv1_observed_rate"] = work.SIGNED_VOLUME_1.notna().groupby(work.ALLOCATION).mean()
    profile["sv1_abs_mean"] = work.SIGNED_VOLUME_1.abs().groupby(work.ALLOCATION).mean()
    ret1_rank = work.RET_1.groupby(work.TS).rank(pct=True)
    turnover_rank = work.MEDIAN_DAILY_TURNOVER.groupby(work.TS).rank(pct=True)
    profile["ret1_rank_mean"] = ret1_rank.groupby(work.ALLOCATION).mean()
    profile["ret1_rank_std"] = ret1_rank.groupby(work.ALLOCATION).std()
    profile["turnover_rank_mean"] = turnover_rank.groupby(work.ALLOCATION).mean()

    date_factor = work.RET_1.groupby(work.TS).transform("mean")
    relation_rows = []
    for allocation, positions in work.groupby("ALLOCATION", observed=True).indices.items():
        positions = np.asarray(positions)
        relation_rows.append(
            {
                "ALLOCATION": allocation,
                "ret1_ret2_corr": safe_corr(
                    work.iloc[positions].RET_1.to_numpy(), work.iloc[positions].RET_2.to_numpy()
                ),
                "ret1_ret3_corr": safe_corr(
                    work.iloc[positions].RET_1.to_numpy(), work.iloc[positions].RET_3.to_numpy()
                ),
                "ret1_date_factor_corr": safe_corr(
                    work.iloc[positions].RET_1.to_numpy(), date_factor.iloc[positions].to_numpy()
                ),
                "ret1_sv1_corr": safe_corr(
                    work.iloc[positions].RET_1.to_numpy(),
                    work.iloc[positions].SIGNED_VOLUME_1.to_numpy(),
                ),
            }
        )
    relations = pd.DataFrame(relation_rows).set_index("ALLOCATION")
    profile = profile.join(relations).sort_index()
    group_label = group.GROUP.agg(lambda x: x.mode().iloc[0]).reindex(profile.index).astype(str)
    return profile, group_label


def learn_archetypes(
    X_train: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, make_pipeline]:
    profile, groups = allocation_behavior_profile(X_train)
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PCA(n_components=min(10, len(profile.columns)), random_state=0),
    )
    coordinates = pipeline.fit_transform(profile)
    kmeans = KMeans(n_clusters=N_ARCHETYPES, n_init=20, random_state=0)
    labels = pd.Series(kmeans.fit_predict(coordinates), index=profile.index, name="archetype")
    standardized = pd.DataFrame(
        StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(profile)),
        index=profile.index,
        columns=profile.columns,
    )
    centroids = standardized.assign(archetype=labels).groupby("archetype").mean()
    return labels, profile, centroids, groups, pipeline


def safe_pair_correlation(frame: pd.DataFrame, left: str, right: str) -> float:
    return safe_corr(frame[left].to_numpy(float), frame[right].to_numpy(float))


def relational_core_profile(X: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    return_pairs = list(combinations(BASE_RETURNS, 2))
    for date, positions in X.groupby("TS", observed=True).indices.items():
        block = X.iloc[np.asarray(positions)]
        row: dict[str, object] = {"TS": str(date)}
        matrix = block[BASE_RETURNS].to_numpy(float)
        corr = np.corrcoef(matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        for left, right in return_pairs:
            i, j = BASE_RETURNS.index(left), BASE_RETURNS.index(right)
            row[f"corr_{left}_{right}"] = corr[i, j]
        eigenvalues = np.linalg.eigvalsh(corr)[::-1]
        eigenvalues = np.clip(eigenvalues, 0, None)
        for index, value in enumerate(eigenvalues, 1):
            row[f"return_corr_eigenvalue_{index}"] = value
        weights = eigenvalues / max(eigenvalues.sum(), 1e-12)
        row["return_corr_effective_rank"] = float(
            np.exp(-np.sum(weights * np.log(np.clip(weights, 1e-12, None))))
        )
        upper = corr[np.triu_indices_from(corr, k=1)]
        row["return_mean_abs_correlation"] = float(np.mean(np.abs(upper)))
        row["return_mean_correlation"] = float(np.mean(upper))
        for feature in BASE_RETURNS:
            lag = feature.split("_")[1]
            row[f"corr_{feature}_turnover"] = safe_pair_correlation(
                block, feature, "MEDIAN_DAILY_TURNOVER"
            )
            volume = f"SIGNED_VOLUME_{lag}"
            row[f"corr_{feature}_{volume}"] = safe_pair_correlation(block, feature, volume)
        rows.append(row)
    return pd.DataFrame(rows).set_index("TS")


def categorical_date_profile(
    X: pd.DataFrame,
    category: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    work = X[["TS"] + BASE_RETURNS + ["MEDIAN_DAILY_TURNOVER", "SIGNED_VOLUME_1"]].copy()
    work["category"] = category.reindex(X.index).fillna("UNKNOWN").astype(str)
    work["sv1_observed"] = work.SIGNED_VOLUME_1.notna().astype(float)
    grouped = work.groupby(["TS", "category"], observed=True)
    parts = []
    for feature in BASE_RETURNS:
        table = grouped[feature].agg(["mean", "std"]).unstack("category")
        table.columns = [f"{prefix}_{category}_{feature}_{stat}" for stat, category in table.columns]
        parts.append(table)
    counts = grouped.size().unstack("category", fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.columns = [f"{prefix}_{column}_share" for column in shares.columns]
    parts.append(shares)
    turnover = grouped.MEDIAN_DAILY_TURNOVER.median().unstack("category")
    turnover.columns = [f"{prefix}_{column}_turnover_median" for column in turnover.columns]
    parts.append(turnover)
    observed = grouped.sv1_observed.mean().unstack("category")
    observed.columns = [f"{prefix}_{column}_sv1_observed" for column in observed.columns]
    parts.append(observed)
    result = pd.concat(parts, axis=1)
    result.index = result.index.astype(str)
    return result


def static_date_profile(X: pd.DataFrame) -> pd.DataFrame:
    marginal = date_profile(X)
    marginal.index = marginal.index.astype(str)
    relational = relational_core_profile(X)
    provided_group = categorical_date_profile(X, X.GROUP.astype(str), "group")
    return marginal.join(relational, how="outer").join(provided_group, how="outer")


def archetype_date_profile(X: pd.DataFrame, assignments: pd.Series) -> tuple[pd.DataFrame, int]:
    row_archetype = X.ALLOCATION.map(assignments)
    unknown = int(row_archetype.isna().sum())
    result = categorical_date_profile(X, row_archetype.astype("Int64").astype(str), "archetype")
    return result, unknown


def model_fold(
    profiles: pd.DataFrame,
    targets: pd.DataFrame,
    train_dates: pd.Index,
    valid_dates: pd.Index,
    parameters: dict[str, tuple[float, float]],
    fold: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    preprocessing = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X_train = preprocessing.fit_transform(profiles.loc[train_dates])
    X_valid = preprocessing.transform(profiles.loc[valid_dates])
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train, targets.loc[train_dates])
    ridge_train = ridge.predict(X_train)
    ridge_valid = ridge.predict(X_valid)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for position, feature in enumerate(BASE_RETURNS):
        global_mean, _ = parameters[feature]
        actual_train = targets.loc[train_dates, feature].to_numpy(float)
        actual_valid = targets.loc[valid_dates, feature].to_numpy(float)
        inversion_train = (actual_train * global_mean < 0).astype(int)
        inversion_valid = (actual_valid * global_mean < 0).astype(int)
        classifier = LogisticRegression(
            C=0.01,
            penalty="l2",
            solver="lbfgs",
            max_iter=300,
            class_weight="balanced",
            random_state=0,
        )
        classifier.fit(X_train, inversion_train)
        p_train = classifier.predict_proba(X_train)[:, 1]
        p_valid = classifier.predict_proba(X_valid)[:, 1]
        train_auc = roc_auc_score(inversion_train, p_train)
        valid_auc = roc_auc_score(inversion_valid, p_valid)
        metric_rows.append(
            {
                "fold": fold,
                "feature": feature,
                "n_train_dates": len(train_dates),
                "n_valid_dates": len(valid_dates),
                "ridge_train_correlation": safe_corr(ridge_train[:, position], actual_train),
                "ridge_valid_correlation": safe_corr(ridge_valid[:, position], actual_valid),
                "ridge_valid_mae": mean_absolute_error(actual_valid, ridge_valid[:, position]),
                "constant_valid_mae": mean_absolute_error(
                    actual_valid, np.full(len(actual_valid), global_mean)
                ),
                "logit_train_auc": train_auc,
                "logit_valid_auc": valid_auc,
                "logit_valid_balanced_accuracy": balanced_accuracy_score(
                    inversion_valid, p_valid >= 0.5
                ),
                "inversion_rate": inversion_valid.mean(),
            }
        )
        for date, actual, predicted, probability, inversion in zip(
            valid_dates,
            actual_valid,
            ridge_valid[:, position],
            p_valid,
            inversion_valid,
        ):
            prediction_rows.append(
                {
                    "TS": date,
                    "fold": fold,
                    "feature": feature,
                    "actual_shrunk_z": actual,
                    "ridge_predicted_z": predicted,
                    "inversion": inversion,
                    "logit_inversion_probability": probability,
                    "global_fisher_z": global_mean,
                }
            )
    return prediction_rows, metric_rows


def summarize(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for feature, block in predictions.groupby("feature", observed=True):
        local_folds = fold_metrics[fold_metrics.feature.eq(feature)]
        train_auc = local_folds.logit_train_auc.mean()
        valid_auc = roc_auc_score(block.inversion, block.logit_inversion_probability)
        ratio = (
            (valid_auc - 0.5) / (train_auc - 0.5)
            if train_auc > 0.500001 else np.nan
        )
        rows.append(
            {
                "feature": feature,
                "n_dates": len(block),
                "ridge_correlation": safe_corr(
                    block.ridge_predicted_z.to_numpy(), block.actual_shrunk_z.to_numpy()
                ),
                "ridge_r2": r2_score(block.actual_shrunk_z, block.ridge_predicted_z),
                "ridge_mae": mean_absolute_error(block.actual_shrunk_z, block.ridge_predicted_z),
                "constant_mae": mean_absolute_error(block.actual_shrunk_z, block.global_fisher_z),
                "mae_improvement_folds": int(
                    local_folds.ridge_valid_mae.lt(local_folds.constant_valid_mae).sum()
                ),
                "logit_train_auc_mean": train_auc,
                "logit_valid_auc": valid_auc,
                "logit_valid_balanced_accuracy": balanced_accuracy_score(
                    block.inversion, block.logit_inversion_probability.ge(0.5)
                ),
                "excess_auc_generalization_ratio": ratio,
            }
        )
    summary = pd.DataFrame(rows)
    gate = summary.assign(
        gate_passed=lambda d: (
            d.ridge_correlation.gt(0.10)
            & d.logit_valid_auc.gt(0.55)
            & d.mae_improvement_folds.ge(6)
            & d.excess_auc_generalization_ratio.ge(0.70)
        )
    )
    return summary, gate[[
        "feature", "ridge_correlation", "logit_valid_auc", "mae_improvement_folds",
        "excess_auc_generalization_ratio", "gate_passed"
    ]]


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train().target
    if not X.index.equals(y.index):
        raise ValueError("X_train et y_train ne sont pas alignés")
    slopes = compute_date_slopes(X, y)
    static = static_date_profile(X)
    dates = pd.Index(X.TS.unique().astype(str))
    static = static.reindex(dates)
    splitter = KFold(n_splits=8, shuffle=True, random_state=0)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    archetype_fold_rows: list[dict[str, object]] = []
    for fold, (train_positions, valid_positions) in enumerate(splitter.split(dates), 1):
        train_dates = dates[train_positions]
        valid_dates = dates[valid_positions]
        train_rows = X.TS.astype(str).isin(train_dates)
        assignments, _, _, group_labels, _ = learn_archetypes(X.loc[train_rows])
        archetype_profile, unknown_rows = archetype_date_profile(X, assignments)
        profiles = static.join(archetype_profile, how="left")
        targets, parameters = build_target_matrix(slopes, dates, train_dates)
        predictions, metrics = model_fold(
            profiles, targets, train_dates, valid_dates, parameters, fold
        )
        prediction_rows.extend(predictions)
        metric_rows.extend(metrics)
        assignment_rows.extend(
            {
                "fold": fold,
                "ALLOCATION": allocation,
                "archetype": int(archetype),
                "GROUP": group_labels.loc[allocation],
            }
            for allocation, archetype in assignments.items()
        )
        archetype_fold_rows.append(
            {
                "fold": fold,
                "n_profile_features": profiles.shape[1],
                "n_allocations": len(assignments),
                "unknown_rows": unknown_rows,
                "adjusted_mutual_information_with_GROUP": adjusted_mutual_info_score(
                    group_labels, assignments
                ),
            }
        )
        print(
            f"H9 fold={fold} features={profiles.shape[1]} allocations={len(assignments)} "
            f"unknown_rows={unknown_rows}",
            flush=True,
        )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(metric_rows)
    assignments = pd.DataFrame(assignment_rows)
    archetype_folds = pd.DataFrame(archetype_fold_rows)
    feature_metrics, gate = summarize(predictions, fold_metrics)

    ari_rows = []
    assignment_wide = assignments.pivot(index="ALLOCATION", columns="fold", values="archetype")
    for left, right in combinations(assignment_wide.columns, 2):
        valid = assignment_wide[[left, right]].dropna()
        ari_rows.append(
            {
                "fold_left": left,
                "fold_right": right,
                "n_allocations": len(valid),
                "adjusted_rand_index": adjusted_rand_score(valid[left], valid[right]),
            }
        )
    ari = pd.DataFrame(ari_rows)

    global_assignments, allocation_profiles, centroids, global_groups, _ = learn_archetypes(X)
    global_membership = pd.DataFrame(
        {
            "ALLOCATION": global_assignments.index,
            "archetype": global_assignments.values,
            "GROUP": global_groups.reindex(global_assignments.index).values,
        }
    )
    group_centroids = (
        pd.DataFrame(
            StandardScaler().fit_transform(
                SimpleImputer(strategy="median").fit_transform(allocation_profiles)
            ),
            index=allocation_profiles.index,
            columns=allocation_profiles.columns,
        )
        .assign(GROUP=global_groups)
        .groupby("GROUP")
        .mean()
    )

    h8_metrics = pd.read_csv(H8_METRICS).rename(
        columns={
            "correlation": "h8_marginal_ridge_correlation",
            "inversion_auc": "h8_marginal_inversion_auc",
        }
    )
    comparison = feature_metrics.merge(
        h8_metrics[["feature", "h8_marginal_ridge_correlation", "h8_marginal_inversion_auc"]],
        on="feature",
        how="left",
    )

    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    feature_metrics.to_csv(output_dir / "feature_metrics.csv", index=False)
    gate.to_csv(output_dir / "gate.csv", index=False)
    comparison.to_csv(output_dir / "h8_h9_comparison.csv", index=False)
    assignments.to_csv(output_dir / "archetype_assignments_by_fold.csv", index=False)
    archetype_folds.to_csv(output_dir / "archetype_fold_diagnostics.csv", index=False)
    ari.to_csv(output_dir / "archetype_stability.csv", index=False)
    allocation_profiles.to_csv(output_dir / "allocation_behavior_profiles.csv")
    global_membership.to_csv(output_dir / "global_archetype_membership.csv", index=False)
    centroids.to_csv(output_dir / "archetype_centroids_z.csv")
    group_centroids.to_csv(output_dir / "group_centroids_z.csv")
    summary = {
        "n_dates": len(dates),
        "n_static_profile_features": int(static.shape[1]),
        "n_archetypes": N_ARCHETYPES,
        "mean_archetype_ari_between_folds": float(ari.adjusted_rand_index.mean()),
        "mean_archetype_ami_with_group": float(
            archetype_folds.adjusted_mutual_information_with_GROUP.mean()
        ),
        "unknown_rows_total": int(archetype_folds.unknown_rows.sum()),
        "n_gate_passed": int(gate.gate_passed.sum()),
        "feature_metrics": feature_metrics.to_dict("records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print("\nFEATURE METRICS")
    print(comparison.to_string(index=False))
    print("\nGATE")
    print(gate.to_string(index=False))
    print("\nARCHETYPES")
    print(json.dumps({key: summary[key] for key in [
        "mean_archetype_ari_between_folds", "mean_archetype_ami_with_group",
        "unknown_rows_total", "n_gate_passed"
    ]}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_dir)


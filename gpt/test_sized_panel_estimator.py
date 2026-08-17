"""H5: panels de 120 dates appariés au test et diagnostic de |score|."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.adversarial_stress_validation import date_profile  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "test_sized_panels"
OOF_PATHS = {
    "V1 Ridge": GPT_DIR / "outputs" / "v1_recheck" / "oof_predictions.csv",
    "V2 logistic": GPT_DIR
    / "outputs"
    / "classification_full"
    / "oof_logistic_C_0.003.csv",
    "LightGBM linear": GPT_DIR
    / "outputs"
    / "v3_lgbm_linear_full8"
    / "oof_lgbm_linear_returns_compact_full.csv",
}
PUBLIC_SCORES = {
    "V1 Ridge": 0.5118293065578914,
    "V2 logistic": 0.5070599309695638,
    "LightGBM linear": 0.5120175713837465,
}


def profile_coordinates(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_profile = date_profile(X_train).add_prefix("")
    test_profile = date_profile(X_test).add_prefix("")
    columns = train_profile.columns.union(test_profile.columns)
    train_profile = train_profile.reindex(columns=columns)
    test_profile = test_profile.reindex(columns=columns)
    combined = pd.concat([train_profile, test_profile], axis=0)
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PCA(n_components=min(n_components, len(columns)), random_state=0),
    )
    coordinates = pipeline.fit_transform(combined)
    raw_scaled = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    ).fit_transform(combined)
    n_train = len(train_profile)
    train_coord = pd.DataFrame(
        coordinates[:n_train], index=train_profile.index,
        columns=[f"PC_{i+1}" for i in range(coordinates.shape[1])],
    )
    test_coord = pd.DataFrame(
        coordinates[n_train:], index=test_profile.index, columns=train_coord.columns
    )
    train_scaled = pd.DataFrame(raw_scaled[:n_train], index=train_profile.index, columns=columns)
    test_scaled = pd.DataFrame(raw_scaled[n_train:], index=test_profile.index, columns=columns)
    return train_coord, test_coord, train_scaled, test_scaled


def nearest_neighbor_map(
    train_coord: pd.DataFrame,
    test_coord: pd.DataFrame,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    search = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    search.fit(train_coord)
    distances, indices = search.kneighbors(test_coord)
    return distances, indices


def matched_panel(
    neighbor_distances: np.ndarray,
    neighbor_indices: np.ndarray,
    n_train_dates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    used: set[int] = set()
    selected: list[int] = []
    global_temperature = float(np.median(neighbor_distances[:, -1])) + 1e-8
    for test_position in rng.permutation(len(neighbor_indices)):
        candidates = neighbor_indices[test_position]
        distances = neighbor_distances[test_position]
        available = np.array([idx not in used for idx in candidates])
        if available.any():
            candidates = candidates[available]
            distances = distances[available]
            weights = np.exp(-(distances - distances.min()) / global_temperature)
            weights = weights / weights.sum()
            choice = int(rng.choice(candidates, p=weights))
        else:
            remaining = np.setdiff1d(
                np.arange(n_train_dates), np.fromiter(used, dtype=int), assume_unique=False
            )
            choice = int(rng.choice(remaining))
        used.add(choice)
        selected.append(choice)
    return np.asarray(selected, dtype=int)


def load_date_metrics() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    date_metrics: dict[str, pd.DataFrame] = {}
    common_rows = None
    for model, path in OOF_PATHS.items():
        oof = pd.read_csv(path, index_col="ROW_ID")
        if common_rows is None:
            common_rows = oof[["fold", "TS", "y_true", "y_true_binarized"]].copy()
        correct = oof.prediction.astype(int).eq(oof.y_true_binarized.astype(int))
        positive = oof.prediction.astype(int)
        table = pd.DataFrame(
            {"TS": oof.TS, "correct": correct.astype(int), "positive": positive}
        ).groupby("TS", observed=True).agg(
            n_rows=("correct", "size"),
            n_correct=("correct", "sum"),
            n_positive=("positive", "sum"),
            date_accuracy=("correct", "mean"),
        )
        date_metrics[model] = table
    assert common_rows is not None
    return date_metrics, common_rows


def panel_model_metrics(table: pd.DataFrame, dates: pd.Index) -> dict[str, float]:
    selected = table.loc[dates]
    return {
        "accuracy": float(selected.n_correct.sum() / selected.n_rows.sum()),
        "date_accuracy": float(selected.date_accuracy.mean()),
        "positive_rate": float(selected.n_positive.sum() / selected.n_rows.sum()),
        "n_rows": int(selected.n_rows.sum()),
    }


def build_panels(
    train_coord: pd.DataFrame,
    test_coord: pd.DataFrame,
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    date_metrics: dict[str, pd.DataFrame],
    n_panels: int,
    n_neighbors: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    distances, indices = nearest_neighbor_map(train_coord, test_coord, n_neighbors)
    train_dates = train_coord.index
    test_mean_coord = test_coord.mean(axis=0).to_numpy()
    test_mean_scaled = test_scaled.mean(axis=0).to_numpy()
    panel_rows = []
    metric_rows = []
    membership_rows = []
    for regime in ["uniform", "matched"]:
        for panel_id in range(1, n_panels + 1):
            if regime == "uniform":
                selected_positions = rng.choice(
                    len(train_dates), size=len(test_coord), replace=False
                )
            else:
                selected_positions = matched_panel(
                    distances, indices, len(train_dates), rng
                )
            selected_dates = train_dates[selected_positions]
            coord_gap = float(
                np.linalg.norm(
                    train_coord.iloc[selected_positions].mean(axis=0).to_numpy()
                    - test_mean_coord
                )
            )
            raw_gap = np.abs(
                train_scaled.iloc[selected_positions].mean(axis=0).to_numpy()
                - test_mean_scaled
            )
            panel_rows.append(
                {
                    "regime": regime,
                    "panel": panel_id,
                    "profile_pca_mean_distance": coord_gap,
                    "profile_mean_abs_smd": float(raw_gap.mean()),
                    "profile_max_abs_smd": float(raw_gap.max()),
                }
            )
            membership_rows.extend(
                {"regime": regime, "panel": panel_id, "TS": date}
                for date in selected_dates
            )
            for model, table in date_metrics.items():
                metric_rows.append(
                    {
                        "regime": regime,
                        "panel": panel_id,
                        "model": model,
                        **panel_model_metrics(table, selected_dates),
                    }
                )
    return pd.DataFrame(panel_rows), pd.DataFrame(metric_rows), pd.DataFrame(membership_rows)


def summarize_panels(panel_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = (
        panel_metrics.groupby(["regime", "model"], observed=True)
        .accuracy.agg(
            mean="mean",
            std="std",
            q025=lambda x: x.quantile(0.025),
            q10=lambda x: x.quantile(0.10),
            median="median",
            q90=lambda x: x.quantile(0.90),
            q975=lambda x: x.quantile(0.975),
        )
        .reset_index()
    )
    wide = panel_metrics.pivot(index=["regime", "panel"], columns="model", values="accuracy")
    difference_rows = []
    public_order = [name for name, _ in sorted(PUBLIC_SCORES.items(), key=lambda x: x[1], reverse=True)]
    for regime, block in wide.groupby(level="regime"):
        local = block.droplevel("regime")
        for comparison, left, right in [
            ("LightGBM linear − V1 Ridge", "LightGBM linear", "V1 Ridge"),
            ("V2 logistic − V1 Ridge", "V2 logistic", "V1 Ridge"),
        ]:
            delta = local[left] - local[right]
            difference_rows.append(
                {
                    "regime": regime,
                    "comparison": comparison,
                    "mean": delta.mean(),
                    "std": delta.std(),
                    "q025": delta.quantile(0.025),
                    "q10": delta.quantile(0.10),
                    "median": delta.median(),
                    "q90": delta.quantile(0.90),
                    "q975": delta.quantile(0.975),
                    "probability_positive": delta.gt(0).mean(),
                }
            )
        rank_match = local.apply(
            lambda row: list(row.sort_values(ascending=False).index) == public_order, axis=1
        )
        rank_corr = local.apply(
            lambda row: spearmanr(
                [row[name] for name in PUBLIC_SCORES],
                [PUBLIC_SCORES[name] for name in PUBLIC_SCORES],
            ).statistic,
            axis=1,
        )
        difference_rows.append(
            {
                "regime": regime,
                "comparison": "classement public exact",
                "mean": rank_match.mean(),
                "std": rank_match.std(),
                "q025": np.nan,
                "q10": np.nan,
                "median": rank_match.median(),
                "q90": np.nan,
                "q975": np.nan,
                "probability_positive": rank_match.mean(),
            }
        )
        difference_rows.append(
            {
                "regime": regime,
                "comparison": "Spearman panel/public",
                "mean": rank_corr.mean(),
                "std": rank_corr.std(),
                "q025": rank_corr.quantile(0.025),
                "q10": rank_corr.quantile(0.10),
                "median": rank_corr.median(),
                "q90": rank_corr.quantile(0.90),
                "q975": rank_corr.quantile(0.975),
                "probability_positive": rank_corr.gt(0).mean(),
            }
        )
    return summaries, pd.DataFrame(difference_rows)


def confidence_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    curve_rows = []
    fold_rows = []
    frames = {}
    for model in ["V1 Ridge", "LightGBM linear"]:
        oof = pd.read_csv(OOF_PATHS[model], index_col="ROW_ID")
        oof["confidence_percentile"] = oof.groupby("fold", observed=True).score.transform(
            lambda score: score.abs().rank(method="average", pct=True)
        )
        oof["confidence_decile"] = np.minimum(
            (10 * oof.confidence_percentile).astype(int), 9
        )
        oof["is_correct"] = oof.prediction.astype(int).eq(
            oof.y_true_binarized.astype(int)
        ).astype(int)
        frames[model] = oof
        grouped = oof.groupby("confidence_decile", observed=True)
        for decile, block in grouped:
            curve_rows.append(
                {
                    "model": model,
                    "confidence_decile": int(decile) + 1,
                    "n_rows": len(block),
                    "accuracy": block.is_correct.mean(),
                    "mean_abs_score": block.score.abs().mean(),
                    "mean_abs_target": block.y_true.abs().mean(),
                    "prediction_positive_rate": block.prediction.mean(),
                }
            )
        for fold, block in oof.groupby("fold", observed=True):
            by_decile = block.groupby("confidence_decile").is_correct.mean()
            slope = float(np.polyfit(by_decile.index, by_decile.values, 1)[0])
            fold_rows.append(
                {
                    "model": model,
                    "fold": int(fold),
                    "slope_accuracy_per_decile": slope,
                    "bottom_decile_accuracy": by_decile.iloc[0],
                    "top_decile_accuracy": by_decile.iloc[-1],
                    "top_minus_bottom": by_decile.iloc[-1] - by_decile.iloc[0],
                }
            )

    v1 = frames["V1 Ridge"]
    lgbm = frames["LightGBM linear"]
    common = v1.index.intersection(lgbm.index)
    disagreement = v1.loc[common, "prediction"].astype(int).ne(
        lgbm.loc[common, "prediction"].astype(int)
    )
    choose_lgbm = lgbm.loc[common, "confidence_percentile"].gt(
        v1.loc[common, "confidence_percentile"]
    )
    selected_prediction = np.where(
        choose_lgbm,
        lgbm.loc[common, "prediction"].astype(int),
        v1.loc[common, "prediction"].astype(int),
    )
    truth = v1.loc[common, "y_true_binarized"].astype(int).to_numpy()
    selector = {
        "overall_accuracy": float(np.mean(selected_prediction == truth)),
        "disagreement_accuracy": float(
            np.mean(selected_prediction[disagreement.to_numpy()] == truth[disagreement.to_numpy()])
        ),
        "disagreement_rate": float(disagreement.mean()),
        "lgbm_chosen_rate": float(choose_lgbm.mean()),
        "v1_accuracy": float(v1.loc[common, "is_correct"].mean()),
        "lgbm_accuracy": float(lgbm.loc[common, "is_correct"].mean()),
    }
    return pd.DataFrame(curve_rows), pd.DataFrame(fold_rows), selector


def run(
    output_dir: Path,
    n_panels: int,
    n_neighbors: int,
    n_components: int,
    seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train = ChallengeDataLoader.load_X_train()
    X_test = ChallengeDataLoader.load_X_test()
    train_coord, test_coord, train_scaled, test_scaled = profile_coordinates(
        X_train, X_test, n_components
    )
    date_metrics, _ = load_date_metrics()
    panel_balance, panel_metrics, memberships = build_panels(
        train_coord,
        test_coord,
        train_scaled,
        test_scaled,
        date_metrics,
        n_panels,
        n_neighbors,
        seed,
    )
    panel_summary, paired_summary = summarize_panels(panel_metrics)
    confidence_curve, confidence_folds, selector = confidence_diagnostics()

    panel_balance.to_csv(output_dir / "panel_balance.csv", index=False)
    panel_metrics.to_csv(output_dir / "panel_metrics.csv", index=False)
    memberships.to_csv(output_dir / "panel_memberships.csv", index=False)
    panel_summary.to_csv(output_dir / "panel_summary.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_summary.csv", index=False)
    confidence_curve.to_csv(output_dir / "confidence_curve.csv", index=False)
    confidence_folds.to_csv(output_dir / "confidence_folds.csv", index=False)
    summary = {
        "n_panels_per_regime": n_panels,
        "n_test_dates_per_panel": len(test_coord),
        "n_neighbors": n_neighbors,
        "n_profile_components": n_components,
        "seed": seed,
        "panel_balance": panel_balance.groupby("regime").agg(
            mean_pca_distance=("profile_pca_mean_distance", "mean"),
            mean_abs_smd=("profile_mean_abs_smd", "mean"),
            mean_max_abs_smd=("profile_max_abs_smd", "mean"),
        ).to_dict("index"),
        "paired_results": paired_summary.to_dict("records"),
        "confidence_positive_slope_folds": confidence_folds.groupby("model")[
            "slope_accuracy_per_decile"
        ].apply(lambda x: int(x.gt(0).sum())).to_dict(),
        "confidence_selector": selector,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(panel_summary.to_string(index=False))
    print("\nPAIRED")
    print(paired_summary.to_string(index=False))
    print("\nCONFIDENCE")
    print(confidence_curve.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-panels", type=int, default=1000)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--n-components", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_dir,
        n_panels=args.n_panels,
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        seed=args.seed,
    )

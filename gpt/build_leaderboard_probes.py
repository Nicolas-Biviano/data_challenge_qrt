"""Construit trois soumissions diagnostiques réellement différentes de V1/V2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.lightgbm_v3 import make_model  # noqa: E402
from gpt.v3_features import BASE_RETURNS, allocation_clusters, build_v3_features  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "leaderboard_probes"


def save_submission(sample: pd.DataFrame, prediction: np.ndarray, path: Path) -> None:
    submission = sample.copy()
    submission.iloc[:, 0] = np.asarray(prediction, dtype="int8")
    if not submission.index.equals(sample.index):
        raise ValueError("L'ordre des ROW_ID a changé")
    if submission.isna().any().any():
        raise ValueError(f"Valeurs manquantes dans {path.name}")
    if not set(submission.iloc[:, 0].unique()).issubset({0, 1}):
        raise ValueError(f"Prédictions non binaires dans {path.name}")
    submission.to_csv(path)


def ridge_returns_only(
    X_train: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    unique_dates = X_train.TS.unique()
    splitter = KFold(n_splits=8, shuffle=True, random_state=0)
    oof_score = pd.Series(np.nan, index=X_train.index)
    for train_date_idx, valid_date_idx in splitter.split(unique_dates):
        train_mask = X_train.TS.isin(unique_dates[train_date_idx])
        valid_mask = X_train.TS.isin(unique_dates[valid_date_idx])
        transform = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        train_matrix = transform.fit_transform(X_train.loc[train_mask, BASE_RETURNS])
        valid_matrix = transform.transform(X_train.loc[valid_mask, BASE_RETURNS])
        model = Ridge(alpha=100.0, solver="lsqr")
        model.fit(train_matrix, y.loc[train_mask, "target"])
        oof_score.loc[valid_mask] = model.predict(valid_matrix)

    oof_prediction = oof_score.gt(0).astype("int8")
    truth = y.target_binarized.astype("int8")
    transform = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    train_matrix = transform.fit_transform(X_train[BASE_RETURNS])
    test_matrix = transform.transform(X_test[BASE_RETURNS])
    model = Ridge(alpha=100.0, solver="lsqr")
    model.fit(train_matrix, y.target)
    test_score = model.predict(test_matrix)
    metrics = {
        "oof_accuracy": float(oof_prediction.eq(truth).mean()),
        "oof_positive_rate": float(oof_prediction.mean()),
        "test_positive_rate": float((test_score > 0).mean()),
    }
    return (test_score > 0).astype("int8"), test_score, metrics


def category_codes(train: pd.Series, test: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    categories = pd.Index(train.dropna().unique())
    return (
        pd.Categorical(train, categories=categories).codes.astype("int32"),
        pd.Categorical(test, categories=categories).codes.astype("int32"),
    )


def lgbm_linear_returns_compact(
    X_train: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_features, blocks = build_v3_features(X_train)
    test_features, _ = build_v3_features(X_test)
    numeric_columns = blocks["returns_compact"]
    train_numeric = train_features[numeric_columns].to_numpy("float32", copy=True)
    test_numeric = test_features[numeric_columns].to_numpy("float32", copy=True)
    means = np.nanmean(train_numeric, axis=0)
    scales = np.nanstd(train_numeric, axis=0)
    scales[(scales == 0) | ~np.isfinite(scales)] = 1.0
    train_numeric = (train_numeric - means) / scales
    test_numeric = (test_numeric - means) / scales

    train_allocation, test_allocation = category_codes(
        X_train.ALLOCATION, X_test.ALLOCATION
    )
    train_group, test_group = category_codes(X_train.GROUP, X_test.GROUP)
    train_cluster = allocation_clusters(X_train, X_train.ALLOCATION).to_numpy("int32")
    test_cluster = allocation_clusters(X_train, X_test.ALLOCATION).to_numpy("int32")
    train_matrix = np.column_stack(
        [train_numeric, train_allocation, train_group, train_cluster]
    ).astype("float32")
    test_matrix = np.column_stack(
        [test_numeric, test_allocation, test_group, test_cluster]
    ).astype("float32")
    feature_names = numeric_columns + [
        "ALLOCATION",
        "GROUP",
        "ALLOCATION_CLUSTER_X_ONLY",
    ]
    categorical_indices = list(range(len(numeric_columns), len(feature_names)))
    model = make_model("linear", n_estimators=150)
    model.fit(
        train_matrix,
        y.target,
        categorical_feature=categorical_indices,
        feature_name=feature_names,
    )
    score = model.predict(test_matrix)
    existing = pd.read_csv(
        Path(__file__).resolve().parent
        / "outputs"
        / "v3_lgbm_linear_full8"
        / "results.csv"
    ).iloc[0]
    metrics = {
        "oof_accuracy": float(existing.accuracy),
        "oof_positive_rate": float(
            pd.read_csv(
                Path(__file__).resolve().parent
                / "outputs"
                / "v3_lgbm_linear_full8"
                / "oof_lgbm_linear_returns_compact_full.csv"
            ).prediction.mean()
        ),
        "test_positive_rate": float((score > 0).mean()),
    }
    return (score > 0).astype("int8"), score, metrics


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    X_test = ChallengeDataLoader.load_X_test()
    sample = ChallengeDataLoader.load_sample_submission()

    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float | str]] = {}

    predictions["ret1_sign"] = X_test.RET_1.gt(0).astype("int8").to_numpy()
    scores["ret1_sign"] = X_test.RET_1.to_numpy(float)
    train_ret1_prediction = X_train.RET_1.gt(0).astype("int8")
    metrics["ret1_sign"] = {
        "oof_accuracy": float(train_ret1_prediction.eq(y.target_binarized).mean()),
        "oof_positive_rate": float(train_ret1_prediction.mean()),
        "test_positive_rate": float(predictions["ret1_sign"].mean()),
        "role": "baseline sans apprentissage",
    }

    ridge_prediction, ridge_score, ridge_metrics = ridge_returns_only(X_train, y, X_test)
    predictions["ridge_returns_only"] = ridge_prediction
    scores["ridge_returns_only"] = ridge_score
    metrics["ridge_returns_only"] = {
        **ridge_metrics,
        "role": "test des effets catégoriels de V1",
    }

    lgbm_prediction, lgbm_score, lgbm_metrics = lgbm_linear_returns_compact(
        X_train, y, X_test
    )
    predictions["lgbm_linear_returns_compact"] = lgbm_prediction
    scores["lgbm_linear_returns_compact"] = lgbm_score
    metrics["lgbm_linear_returns_compact"] = {
        **lgbm_metrics,
        "role": "modèle le plus divers face à V1/V2",
    }

    v1_test_score = pd.read_csv(
        Path(__file__).resolve().parent
        / "outputs"
        / "v1_recheck"
        / "test_predictions.csv",
        index_col="ROW_ID",
    )["raw_score"].to_numpy()
    blend_score = 0.5 * v1_test_score + 0.5 * lgbm_score
    predictions["blend_v1_lgbm_50_50"] = (blend_score > 0).astype("int8")
    scores["blend_v1_lgbm_50_50"] = blend_score
    metrics["blend_v1_lgbm_50_50"] = {
        "oof_accuracy": 0.5252023913196084,
        "oof_positive_rate": 0.5493736161784041,
        "test_positive_rate": float(predictions["blend_v1_lgbm_50_50"].mean()),
        "role": "blend préfixé des deux meilleurs modèles publics",
    }

    for name, prediction in predictions.items():
        save_submission(sample, prediction, output_dir / f"submission_{name}.csv")
    pd.DataFrame(predictions, index=X_test.index).rename_axis("ROW_ID").to_csv(
        output_dir / "test_predictions.csv"
    )
    pd.DataFrame(scores, index=X_test.index).rename_axis("ROW_ID").to_csv(
        output_dir / "test_scores.csv"
    )

    v1 = pd.read_csv(
        Path(__file__).resolve().parent / "outputs" / "v1_recheck" / "submission.csv",
        index_col="ROW_ID",
    ).iloc[:, 0].to_numpy()
    v2 = pd.read_csv(
        Path(__file__).resolve().parent / "outputs" / "v2" / "submission.csv",
        index_col="ROW_ID",
    ).iloc[:, 0].to_numpy()
    all_predictions = {"V1": v1, "V2": v2, **predictions}
    disagreement = pd.DataFrame(
        {
            left: {
                right: float(np.mean(left_prediction != right_prediction))
                for right, right_prediction in all_predictions.items()
            }
            for left, left_prediction in all_predictions.items()
        }
    )
    disagreement.to_csv(output_dir / "test_prediction_disagreement.csv")
    summary = {
        "metrics": metrics,
        "submission_priority": [
            "blend_v1_lgbm_50_50",
            "ridge_returns_only",
            "ret1_sign",
            "lgbm_linear_returns_compact",
        ],
        "known_public_scores": {"V1": "environ 0.5118", "V2": "environ 0.5070"},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nDésaccords test")
    print(disagreement.to_string())
    return summary


if __name__ == "__main__":
    run()

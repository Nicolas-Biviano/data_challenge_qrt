"""Compare CV ordinaire, holdout test-like et leaderboard pour trois modèles reliés."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.lightgbm_v3 import make_model, standardized_matrices  # noqa: E402
from gpt.model import FixedEffectDesign, MODEL_FEATURES  # noqa: E402
from gpt.model_v2 import MODEL_C  # noqa: E402
from gpt.v3_features import allocation_clusters, build_v3_features  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "leaderboard_alignment"
TEST_LIKENESS_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "adversarial_stress_validation"
    / "train_date_test_likeness.csv"
)


def metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = score > threshold
    return {
        "stress_accuracy": float(accuracy_score(truth, prediction)),
        "stress_balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "stress_auc": float(roc_auc_score(truth, score)),
        "stress_positive_rate": float(prediction.mean()),
    }


def run(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    likeness = pd.read_csv(TEST_LIKENESS_PATH, index_col="TS")
    valid_dates = likeness.head(120).index
    train_mask = ~X.TS.isin(valid_dates)
    valid_mask = X.TS.isin(valid_dates)
    truth = y.loc[valid_mask, "target_binarized"].to_numpy("int8")

    design = FixedEffectDesign().fit(X.loc[train_mask, MODEL_FEATURES])
    train_matrix = design.transform(X.loc[train_mask, MODEL_FEATURES])
    valid_matrix = design.transform(X.loc[valid_mask, MODEL_FEATURES])

    ridge = Ridge(alpha=100.0, solver="lsqr")
    ridge.fit(train_matrix, y.loc[train_mask, "target"])
    ridge_score = ridge.predict(valid_matrix)

    logistic = LogisticRegression(
        C=MODEL_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=200,
        tol=1e-5,
        random_state=0,
    )
    logistic.fit(train_matrix, y.loc[train_mask, "target_binarized"])
    logistic_score = logistic.predict_proba(valid_matrix)[:, 1]

    features, blocks = build_v3_features(X)
    allocation_code = pd.Categorical(X.ALLOCATION).codes.astype("int32")
    group_code = pd.Categorical(X.GROUP).codes.astype("int32")
    cluster_code = allocation_clusters(
        X.loc[train_mask], X.ALLOCATION, n_clusters=8
    ).to_numpy()
    lgbm_train, lgbm_valid, feature_names, categorical_indices = standardized_matrices(
        features,
        blocks["returns_compact"],
        train_mask,
        valid_mask,
        allocation_code,
        group_code,
        cluster_code,
        "full",
    )
    lgbm = make_model("linear", n_estimators=150)
    lgbm.fit(
        lgbm_train,
        y.loc[train_mask, "target"],
        categorical_feature=categorical_indices,
        feature_name=feature_names,
    )
    lgbm_score = lgbm.predict(lgbm_valid)

    rows = [
        {
            "model": "V1 Ridge fixed effects",
            "ordinary_oof_accuracy": 0.5247091010163678,
            "public_score": 0.5118293065578914,
            **metrics(truth, ridge_score, 0.0),
        },
        {
            "model": "V2 local C=.003 / public logit C=.001",
            "ordinary_oof_accuracy": 0.525001280657518,
            "public_score": 0.5070599309695638,
            **metrics(truth, logistic_score, 0.5),
        },
        {
            "model": "LightGBM linear returns compact",
            "ordinary_oof_accuracy": 0.5244415858903795,
            "public_score": 0.5120175713837465,
            **metrics(truth, lgbm_score, 0.0),
        },
    ]
    table = pd.DataFrame(rows)
    table["ordinary_rank"] = table.ordinary_oof_accuracy.rank(ascending=False)
    table["stress_rank"] = table.stress_accuracy.rank(ascending=False)
    table["public_rank"] = table.public_score.rank(ascending=False)
    ordinary_corr = float(spearmanr(table.ordinary_oof_accuracy, table.public_score).statistic)
    stress_corr = float(spearmanr(table.stress_accuracy, table.public_score).statistic)
    table.to_csv(output_dir / "alignment_table.csv", index=False)
    summary = {
        "n_stress_dates": int(len(valid_dates)),
        "n_stress_rows": int(valid_mask.sum()),
        "spearman_ordinary_vs_public": ordinary_corr,
        "spearman_stress_vs_public": stress_corr,
        "public_scores_source": "historique communiqué par l'utilisateur le 5 août 2026 ; correspondance logistique approximative",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return table


if __name__ == "__main__":
    run()

"""CV V2 sur les dates completes, proches de la structure dominante du test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.model import FixedEffectDesign  # noqa: E402
from gpt.model_v2 import MODEL_C  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


MODEL_FEATURES = [
    "ALLOCATION",
    "GROUP",
    "RET_1",
    "RET_2",
    "RET_3",
    "RET_4",
    "RET_7",
    "RET_8",
    "RET_9",
    "RET_18",
]


def fit_predict(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    c: float,
) -> np.ndarray:
    """Ajuste la V2 avec preprocessing strictement appris sur le train."""
    design = FixedEffectDesign().fit(X.loc[train_mask, MODEL_FEATURES])
    train_matrix = design.transform(X.loc[train_mask, MODEL_FEATURES])
    valid_matrix = design.transform(X.loc[valid_mask, MODEL_FEATURES])
    model = LogisticRegression(
        C=c,
        penalty="l2",
        solver="lbfgs",
        max_iter=200,
        tol=1e-5,
        random_state=0,
    )
    model.fit(train_matrix, y.loc[train_mask, "target_binarized"])
    return model.predict_proba(valid_matrix)[:, 1]


def summarize_oof(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Resume accuracy, AUC, taux positif et dispersion par date."""
    accuracy = float(frame["is_correct"].mean())
    by_ts = frame.groupby("TS", observed=True)["is_correct"].mean()
    return {
        "training_regime": str(frame["training_regime"].iloc[0]),
        "C": float(frame["C"].iloc[0]),
        "n_rows": int(len(frame)),
        "n_dates": int(frame["TS"].nunique()),
        "accuracy": accuracy,
        "auc": float(roc_auc_score(frame["target"], frame["probability"])),
        "target_positive_rate": float(frame["target"].mean()),
        "prediction_positive_rate": float(frame["prediction"].mean()),
        "ts_standard_error": float(by_ts.std() / np.sqrt(len(by_ts))),
        "ts_penalized_score": float(accuracy - by_ts.std() / np.sqrt(len(by_ts))),
    }


def run(output_dir: Path, regularizations: list[float]) -> pd.DataFrame:
    """Compare train complet et train dates completes sur memes validations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    X_test = ChallengeDataLoader.load_X_test()
    train_date_size = X.groupby("TS", observed=True).size()
    test_date_size = X_test.groupby("TS", observed=True).size()
    dominant_train_size = int(train_date_size.value_counts().index[0])
    dominant_test_size = int(test_date_size.value_counts().index[0])
    matched_dates = train_date_size[train_date_size.eq(dominant_train_size)].index
    splits = KFold(n_splits=8, shuffle=True, random_state=0).split(matched_dates)
    states: dict[tuple[str, float], list[pd.DataFrame]] = {
        ("all_dates", MODEL_C): []
    }
    states.update({("complete_dates_only", c): [] for c in regularizations})
    fold_rows = []

    for fold, (train_index, valid_index) in enumerate(splits, start=1):
        fold_train_dates = matched_dates[train_index]
        valid_dates = matched_dates[valid_index]
        valid_mask = X["TS"].isin(valid_dates)
        train_masks = {
            "all_dates": ~valid_mask,
            "complete_dates_only": X["TS"].isin(fold_train_dates),
        }
        configurations = [("all_dates", MODEL_C)] + [
            ("complete_dates_only", c) for c in regularizations
        ]
        for regime, c in configurations:
            probability = fit_predict(
                X,
                y,
                train_masks[regime],
                valid_mask,
                c,
            )
            truth = y.loc[valid_mask, "target_binarized"].to_numpy(dtype="int8")
            prediction = (probability > 0.5).astype("int8")
            local = pd.DataFrame(
                {
                    "ROW_ID": X.index[valid_mask],
                    "fold": fold,
                    "TS": X.loc[valid_mask, "TS"].to_numpy(),
                    "target": truth,
                    "probability": probability,
                    "prediction": prediction,
                    "is_correct": prediction == truth,
                    "training_regime": regime,
                    "C": c,
                }
            )
            states[(regime, c)].append(local)
            fold_rows.append(
                {
                    "training_regime": regime,
                    "C": c,
                    "fold": fold,
                    "n_train": int(train_masks[regime].sum()),
                    "n_valid": int(valid_mask.sum()),
                    "accuracy": accuracy_score(truth, prediction),
                    "prediction_positive_rate": float(prediction.mean()),
                }
            )
            print(
                f"{regime} C={c:g} fold={fold} "
                f"accuracy={fold_rows[-1]['accuracy']:.6f}",
                flush=True,
            )

    summaries = []
    for key, frames in states.items():
        oof = pd.concat(frames, ignore_index=True).set_index("ROW_ID").sort_index()
        summaries.append(summarize_oof(oof))
        oof.to_csv(output_dir / f"oof_{key[0]}_C_{key[1]:g}.csv")
    result = pd.DataFrame(summaries).sort_values(
        ["accuracy", "ts_penalized_score"], ascending=False
    )
    result["dominant_train_date_size"] = dominant_train_size
    result["dominant_test_date_size"] = dominant_test_size
    result.to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_results.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regularizations", nargs="+", type=float, default=[0.001, 0.003, 0.01])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "test_like_regime_cv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(args.output_dir, args.regularizations).to_string(index=False))

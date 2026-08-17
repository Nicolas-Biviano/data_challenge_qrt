"""Screening d'un modele deux-parties pour SIGNED_VOLUME_1.

La premiere partie modelise la disponibilite du volume. La seconde utilise sa
valeur uniquement lorsqu'elle existe, sous des formes robustes et bornees.
Aucun target encoding et aucun ordre des dates ne sont utilises.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


CATEGORICAL_FEATURES = ["ALLOCATION", "GROUP"]
AVAILABILITY_FEATURES = [
    "sv1_available",
    "sv_available_share_5",
    "sv_available_share_20",
    "date_sv1_available_share",
    "group_date_sv1_available_share",
]
VALUE_FEATURES = [
    "sv1_signed_log",
    "sv1_abs_log",
    "sv1_sign",
]
RANK_FEATURES = [
    "date_rank_sv1",
    "group_date_rank_sv1",
]
RETURN_INTERACTIONS = [
    "ret1_x_sv1_available",
    "ret1_x_date_rank_sv1",
]


FEATURE_SETS: dict[str, list[str]] = {
    "baseline": BASE_RETURNS,
    "presence": BASE_RETURNS + ["sv1_available"],
    "availability_context": BASE_RETURNS + AVAILABILITY_FEATURES,
    "two_part_value": BASE_RETURNS + ["sv1_available"] + VALUE_FEATURES,
    "two_part_rank": BASE_RETURNS + ["sv1_available"] + RANK_FEATURES,
    "two_part_full": (
        BASE_RETURNS
        + AVAILABILITY_FEATURES
        + VALUE_FEATURES
        + RANK_FEATURES
        + RETURN_INTERACTIONS
    ),
}


def build_volume_features(X: pd.DataFrame) -> pd.DataFrame:
    """Construit des variables deux-parties uniquement a partir de X."""
    features = X.copy()
    sv1 = X["SIGNED_VOLUME_1"]
    available = sv1.notna()
    volumes = X[[f"SIGNED_VOLUME_{lag}" for lag in range(1, 21)]]

    features["sv1_available"] = available.astype("float32")
    features["sv_available_share_5"] = volumes.iloc[:, :5].notna().mean(axis=1).astype(
        "float32"
    )
    features["sv_available_share_20"] = volumes.notna().mean(axis=1).astype("float32")
    features["date_sv1_available_share"] = (
        features.groupby("TS", observed=True)["sv1_available"]
        .transform("mean")
        .astype("float32")
    )
    features["group_date_sv1_available_share"] = (
        features.groupby(["TS", "GROUP"], observed=True)["sv1_available"]
        .transform("mean")
        .astype("float32")
    )

    features["sv1_signed_log"] = (
        np.sign(sv1) * np.log1p(sv1.abs())
    ).astype("float32")
    features["sv1_abs_log"] = np.log1p(sv1.abs()).astype("float32")
    features["sv1_sign"] = np.sign(sv1).astype("float32")
    features["date_rank_sv1"] = (
        features.groupby("TS", observed=True)["SIGNED_VOLUME_1"]
        .rank(pct=True, method="average")
        .astype("float32")
    )
    features["group_date_rank_sv1"] = (
        features.groupby(["TS", "GROUP"], observed=True)["SIGNED_VOLUME_1"]
        .rank(pct=True, method="average")
        .astype("float32")
    )
    features["ret1_x_sv1_available"] = (
        features["RET_1"] * features["sv1_available"]
    ).astype("float32")
    features["ret1_x_date_rank_sv1"] = (
        features["RET_1"] * (features["date_rank_sv1"] - 0.5)
    ).astype("float32")
    return features


@dataclass
class VolumeDesign:
    """Design V2 avec interactions X-only optionnelles de disponibilite."""

    numeric_features: list[str]
    availability_interaction: str = "none"

    def __post_init__(self) -> None:
        self.numeric_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
        self.category_encoder = OneHotEncoder(handle_unknown="ignore")
        self.allocation_encoder = OneHotEncoder(handle_unknown="ignore")
        self.group_encoder = OneHotEncoder(handle_unknown="ignore")

    def fit(self, X: pd.DataFrame) -> "VolumeDesign":
        self.numeric_pipeline.fit(X[self.numeric_features])
        self.category_encoder.fit(X[CATEGORICAL_FEATURES])
        self.allocation_encoder.fit(X[["ALLOCATION"]])
        self.group_encoder.fit(X[["GROUP"]])
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        numeric = self.numeric_pipeline.transform(X[self.numeric_features])
        categories = self.category_encoder.transform(X[CATEGORICAL_FEATURES])
        allocation = self.allocation_encoder.transform(X[["ALLOCATION"]])
        ret1_position = self.numeric_features.index("RET_1")
        allocation_ret1 = allocation.multiply(numeric[:, ret1_position, None])
        blocks = [sparse.csr_matrix(numeric), categories, allocation_ret1]

        if self.availability_interaction != "none":
            availability = X["sv1_available"].to_numpy(dtype="float32")[:, None]
            if self.availability_interaction in {"group", "both"}:
                group = self.group_encoder.transform(X[["GROUP"]])
                blocks.append(group.multiply(availability))
            if self.availability_interaction in {"allocation", "both"}:
                blocks.append(allocation.multiply(availability))
        return sparse.hstack(blocks, format="csr")


def experiment_spec(name: str) -> tuple[list[str], str]:
    """Retourne features numeriques et type d'interaction categorielle."""
    if name == "presence_group":
        return FEATURE_SETS["presence"], "group"
    if name == "presence_allocation":
        return FEATURE_SETS["presence"], "allocation"
    if name == "presence_both":
        return FEATURE_SETS["presence"], "both"
    return FEATURE_SETS[name], "none"


EXPERIMENT_NAMES = list(FEATURE_SETS) + [
    "presence_group",
    "presence_allocation",
    "presence_both",
]


def score_summary(
    prediction: pd.Series,
    X: pd.DataFrame,
    y: pd.DataFrame,
) -> dict[str, float | int]:
    """Mesures OOF communes, calculees seulement sur les lignes couvertes."""
    covered = prediction.notna()
    correct = prediction.loc[covered].astype("int8").eq(
        y.loc[covered, "target_binarized"].astype("int8")
    )
    by_ts = correct.groupby(X.loc[covered, "TS"]).mean()
    by_allocation = correct.groupby(X.loc[covered, "ALLOCATION"]).mean()
    accuracy = float(correct.mean())
    return {
        "n_oof_rows": int(covered.sum()),
        "accuracy": accuracy,
        "ts_standard_error": float(by_ts.std() / np.sqrt(len(by_ts))),
        "ts_penalized_score": float(accuracy - by_ts.std() / np.sqrt(len(by_ts))),
        "allocation_standard_error": float(
            by_allocation.std() / np.sqrt(len(by_allocation))
        ),
    }


def run_screen(
    names: list[str],
    regularizations: list[float],
    max_folds: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Evalue les variantes sur des folds de dates fixes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw = ChallengeDataLoader.load_X_train()
    X = build_volume_features(X_raw)
    y = ChallengeDataLoader.load_y_train()
    unique_ts = X["TS"].unique()
    splits = KFold(n_splits=8, shuffle=True, random_state=0).split(unique_ts)
    states: dict[tuple[str, float], dict[str, object]] = {}
    for name in names:
        for c in regularizations:
            states[(name, c)] = {
                "score": pd.Series(np.nan, index=X.index),
                "prediction": pd.Series(np.nan, index=X.index),
                "fold": pd.Series(np.nan, index=X.index),
                "fold_rows": [],
            }

    for fold, (train_date_idx, valid_date_idx) in enumerate(splits, start=1):
        if fold > max_folds:
            break
        train_mask = X["TS"].isin(unique_ts[train_date_idx])
        valid_mask = X["TS"].isin(unique_ts[valid_date_idx])
        for name in names:
            numeric_features, interaction = experiment_spec(name)
            selected = CATEGORICAL_FEATURES + list(
                dict.fromkeys(numeric_features + ["sv1_available"])
            )
            design = VolumeDesign(numeric_features, interaction).fit(
                X.loc[train_mask, selected]
            )
            train_matrix = design.transform(X.loc[train_mask, selected])
            valid_matrix = design.transform(X.loc[valid_mask, selected])
            for c in regularizations:
                model = LogisticRegression(
                    C=c,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=200,
                    tol=1e-5,
                    random_state=0,
                )
                model.fit(train_matrix, y.loc[train_mask, "target_binarized"])
                probability = model.predict_proba(valid_matrix)[:, 1]
                prediction = (probability > 0.5).astype("int8")
                accuracy = accuracy_score(
                    y.loc[valid_mask, "target_binarized"],
                    prediction,
                )
                state = states[(name, c)]
                state["score"].loc[valid_mask] = probability
                state["prediction"].loc[valid_mask] = prediction
                state["fold"].loc[valid_mask] = fold
                state["fold_rows"].append(
                    {
                        "experiment": name,
                        "C": c,
                        "fold": fold,
                        "accuracy": accuracy,
                    }
                )
                print(
                    f"{name} C={c:g} fold={fold} accuracy={accuracy:.6f}",
                    flush=True,
                )

    results = []
    fold_rows = []
    for (name, c), state in states.items():
        row = {
            "experiment": name,
            "C": c,
            "n_folds": len(state["fold_rows"]),
            **score_summary(state["prediction"], X, y),
        }
        results.append(row)
        fold_rows.extend(state["fold_rows"])
        covered = state["prediction"].notna()
        oof = pd.DataFrame(
            {
                "fold": state["fold"],
                "TS": X["TS"],
                "ALLOCATION": X["ALLOCATION"],
                "available": X["sv1_available"],
                "y_true_binarized": y["target_binarized"],
                "score": state["score"],
                "prediction": state["prediction"],
            },
            index=X.index,
        )
        oof["is_correct"] = np.where(
            covered,
            oof["prediction"].eq(oof["y_true_binarized"]).astype(float),
            np.nan,
        )
        oof.index.name = "ROW_ID"
        oof.loc[covered].to_csv(output_dir / f"oof_{name}_C_{c:g}.csv")

    result_table = pd.DataFrame(results).sort_values(
        ["accuracy", "ts_penalized_score"],
        ascending=False,
    )
    result_table.to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_results.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "experiments": names,
                "regularizations": regularizations,
                "max_folds": max_folds,
                "target_encoding": False,
                "date_order_used": False,
            },
            stream,
            indent=2,
        )
    return result_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_NAMES,
        default=EXPERIMENT_NAMES,
    )
    parser.add_argument("--regularizations", nargs="+", type=float, default=[0.003])
    parser.add_argument("--max-folds", type=int, choices=range(1, 9), default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "volume_two_part_screen",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    table = run_screen(
        args.experiments,
        args.regularizations,
        args.max_folds,
        args.output_dir,
    )
    print(table.to_string(index=False))

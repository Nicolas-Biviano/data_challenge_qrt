"""H4: transformations non linéaires X-only et interactions groupées régularisées."""

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.feature_experiments import BASE_RETURNS  # noqa: E402
from gpt.hypotheses_h1_h2_h3 import cluster_bootstrap_models  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


MODEL_C = 0.003
CATEGORICAL = ["ALLOCATION", "GROUP"]
REFERENCE_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "classification_full"
    / "oof_logistic_C_0.003.csv"
)
STAGE_NAMES = {
    "A": "H4-A_global_nonlinear",
    "B": "H4-B_group_interactions",
    "C": "H4-C_allocation_interactions",
    "D": "H4-D_sv1_availability",
    "E": "H4-E_arcsin_tails",
}


def robust_group_z(values: pd.Series, groups: list[pd.Series]) -> pd.Series:
    """Z-score robuste intra-groupe, borné pour éviter les IQR quasi nuls."""
    grouped = values.groupby(groups, observed=True)
    median = grouped.transform("median")
    q25 = grouped.transform(lambda x: x.quantile(.25))
    q75 = grouped.transform(lambda x: x.quantile(.75))
    scale = (q75 - q25) / 1.349
    result = (values - median) / scale.replace(0.0, np.nan)
    return result.clip(-6, 6).fillna(0.0).astype("float32")


def add_basis(features: dict[str, pd.Series], name: str, z: pd.Series) -> list[str]:
    """Ajoute la base H4 préenregistrée et retourne ses noms."""
    names = []
    forms = {
        "linear": z,
        "tanh05": np.tanh(z / .5),
        "tanh1": np.tanh(z),
        "arctan": np.arctan(z),
        "signed_log": np.sign(z) * np.log1p(np.abs(z)),
    }
    for suffix, values in forms.items():
        column = f"{name}__{suffix}"
        features[column] = pd.Series(values, index=z.index, dtype="float32")
        names.append(column)
    return names


def build_h4_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Construit les features H4 uniquement à partir de X."""
    output = X[["TS", "ALLOCATION", "GROUP"] + BASE_RETURNS].copy()
    additions: dict[str, pd.Series] = {}

    ret1_date_z = robust_group_z(X["RET_1"], [X["TS"]])
    ret1_group_z = robust_group_z(X["RET_1"], [X["TS"], X["GROUP"]])
    ret1_date_rank = X.groupby("TS", observed=True)["RET_1"].rank(pct=True).astype("float32")
    ret1_group_rank = X.groupby(["TS", "GROUP"], observed=True)["RET_1"].rank(pct=True).astype("float32")
    turnover_date_rank = X.groupby("TS", observed=True)["MEDIAN_DAILY_TURNOVER"].rank(pct=True).astype("float32")
    turnover_group_rank = X.groupby(["TS", "GROUP"], observed=True)["MEDIAN_DAILY_TURNOVER"].rank(pct=True).astype("float32")

    basis_names: dict[str, list[str]] = {}
    basis_names["ret1_date"] = add_basis(additions, "ret1_date_z", ret1_date_z)
    basis_names["ret1_group"] = add_basis(additions, "ret1_group_z", ret1_group_z)
    for name, rank in {
        "ret1_date_rank": ret1_date_rank,
        "ret1_group_rank": ret1_group_rank,
        "turnover_date_rank": turnover_date_rank,
        "turnover_group_rank": turnover_group_rank,
    }.items():
        centered = (2 * rank - 1).fillna(0.0).astype("float32")
        basis_names[name] = add_basis(additions, name, centered)
        for knot in (.2, .5, .8):
            column = f"{name}__hinge_{int(100*knot)}"
            additions[column] = (rank - knot).clip(lower=0).fillna(0.0).astype("float32")
            basis_names[name].append(column)

    volumes = X[[f"SIGNED_VOLUME_{lag}" for lag in range(1, 21)]]
    additions["sv1_available"] = X["SIGNED_VOLUME_1"].notna().astype("float32")
    sv_positive_share = (
        volumes.gt(0).sum(axis=1) / volumes.notna().sum(axis=1).clip(lower=1)
    ).astype("float32")
    basis_names["sv_positive_share"] = add_basis(
        additions,
        "sv_positive_share_centered",
        (2 * sv_positive_share - 1).astype("float32"),
    )

    output = pd.concat([output, pd.DataFrame(additions, index=X.index)], axis=1)

    nonlinear = [name for names in basis_names.values() for name in names] + ["sv1_available"]
    group_interactions = (
        [name for name in basis_names["ret1_group"] if "linear" not in name]
        + [name for name in basis_names["turnover_group_rank"] if "linear" not in name and "hinge_50" not in name]
        + [basis_names["sv_positive_share"][1]]
    )
    allocation_interactions = [
        name for name in basis_names["ret1_date"] if "linear" not in name
    ]
    sv1_return_interactions = []
    missing = 1.0 - output["sv1_available"]
    for source in allocation_interactions:
        column = f"sv1_missing_x_{source}"
        output[column] = (missing * output[source]).astype("float32")
        sv1_return_interactions.append(column)

    arcsin_features = []
    for name, rank in {
        "ret1_date_rank": ret1_date_rank,
        "ret1_group_rank": ret1_group_rank,
        "turnover_date_rank": turnover_date_rank,
        "turnover_group_rank": turnover_group_rank,
    }.items():
        column = f"{name}__arcsin"
        output[column] = np.arcsin(np.clip(2 * rank - 1, -.999999, .999999)).fillna(0.0).astype("float32")
        arcsin_features.append(column)

    blocks = {
        "nonlinear": nonlinear,
        "group_interactions": group_interactions,
        "allocation_interactions": allocation_interactions,
        "sv1_return_interactions": sv1_return_interactions,
        "arcsin": arcsin_features,
    }
    return output, blocks


@dataclass
class H4Design:
    stage: str
    blocks: dict[str, list[str]]
    group_scale: float = .5
    allocation_scale: float = .25

    def __post_init__(self) -> None:
        self.numeric_features = BASE_RETURNS + list(self.blocks["nonlinear"])
        if self.stage in {"D", "E"}:
            self.numeric_features += list(self.blocks["sv1_return_interactions"])
        if self.stage == "E":
            self.numeric_features += list(self.blocks["arcsin"])
        self.numeric_features = list(dict.fromkeys(self.numeric_features))
        self.numeric_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
        self.category_encoder = OneHotEncoder(handle_unknown="ignore")
        self.allocation_encoder = OneHotEncoder(handle_unknown="ignore")
        self.group_encoder = OneHotEncoder(handle_unknown="ignore")

    def fit(self, X: pd.DataFrame) -> "H4Design":
        self.numeric_pipeline.fit(X[self.numeric_features])
        self.category_encoder.fit(X[CATEGORICAL])
        self.allocation_encoder.fit(X[["ALLOCATION"]])
        self.group_encoder.fit(X[["GROUP"]])
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        numeric = self.numeric_pipeline.transform(X[self.numeric_features])
        categories = self.category_encoder.transform(X[CATEGORICAL])
        allocation = self.allocation_encoder.transform(X[["ALLOCATION"]])
        group = self.group_encoder.transform(X[["GROUP"]])
        ret1_position = self.numeric_features.index("RET_1")
        blocks = [sparse.csr_matrix(numeric), categories, allocation.multiply(numeric[:, ret1_position, None])]

        if self.stage in {"B", "C", "D", "E"}:
            for feature in self.blocks["group_interactions"]:
                position = self.numeric_features.index(feature)
                blocks.append(group.multiply(numeric[:, position, None]) * self.group_scale)
        if self.stage in {"C", "D", "E"}:
            for feature in self.blocks["allocation_interactions"]:
                position = self.numeric_features.index(feature)
                blocks.append(allocation.multiply(numeric[:, position, None]) * self.allocation_scale)
        if self.stage in {"D", "E"}:
            missing = (1.0 - X["sv1_available"].to_numpy(dtype="float32"))[:, None]
            blocks.append(group.multiply(missing) * self.group_scale)
        return sparse.hstack(blocks, format="csr")


def stage_metrics(oof: pd.DataFrame, complete_dates: pd.Index) -> dict[str, float | int]:
    covered = oof.prediction.notna()
    local = oof[covered].copy()
    truth = local.y_true_binarized.astype(int)
    prediction = local.prediction.astype(int)
    complete = local[local.TS.isin(complete_dates)]
    return {
        "n_rows": int(len(local)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "accuracy_complete_dates": float(accuracy_score(complete.y_true_binarized, complete.prediction)),
        "auc": float(roc_auc_score(truth, local.score)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "accuracy_y0": float(accuracy_score(truth[truth.eq(0)], prediction[truth.eq(0)])),
        "accuracy_y1": float(accuracy_score(truth[truth.eq(1)], prediction[truth.eq(1)])),
        "prediction_positive_rate": float(prediction.mean()),
    }


def run_stage(
    stage: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    blocks: dict[str, list[str]],
    output_dir: Path,
    max_folds: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exécute une étape sur les folds fixes et sauvegarde les prédictions OOF."""
    unique_ts = X.TS.unique()
    splitter = KFold(n_splits=8, shuffle=True, random_state=0)
    oof = pd.DataFrame(
        {
            "fold": np.nan,
            "TS": X.TS,
            "ALLOCATION": X.ALLOCATION,
            "y_true": y.target,
            "y_true_binarized": y.target_binarized,
            "score": np.nan,
            "prediction": np.nan,
        },
        index=X.index,
    )
    fold_rows = []
    for fold, (train_date_idx, valid_date_idx) in enumerate(splitter.split(unique_ts), start=1):
        if fold > max_folds:
            break
        train_mask = X.TS.isin(unique_ts[train_date_idx])
        valid_mask = X.TS.isin(unique_ts[valid_date_idx])
        design = H4Design(stage, blocks).fit(X.loc[train_mask])
        train_matrix = design.transform(X.loc[train_mask])
        valid_matrix = design.transform(X.loc[valid_mask])
        model = LogisticRegression(
            C=MODEL_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=200,
            tol=1e-5,
            random_state=0,
        )
        model.fit(train_matrix, y.loc[train_mask, "target_binarized"])
        score = model.predict_proba(valid_matrix)[:, 1]
        prediction = (score > .5).astype("int8")
        truth = y.loc[valid_mask, "target_binarized"].to_numpy(int)
        oof.loc[valid_mask, "fold"] = fold
        oof.loc[valid_mask, "score"] = score
        oof.loc[valid_mask, "prediction"] = prediction
        fold_rows.append(
            {
                "stage": stage,
                "fold": fold,
                "n_train": int(train_mask.sum()),
                "n_valid": int(valid_mask.sum()),
                "n_features": int(train_matrix.shape[1]),
                "n_iter": int(model.n_iter_.max()),
                "accuracy": accuracy_score(truth, prediction),
                "auc": roc_auc_score(truth, score),
                "prediction_positive_rate": prediction.mean(),
            }
        )
        print(
            f"H4-{stage} fold={fold} features={train_matrix.shape[1]} "
            f"accuracy={fold_rows[-1]['accuracy']:.6f} auc={fold_rows[-1]['auc']:.6f}",
            flush=True,
        )
    oof["is_correct"] = np.where(
        oof.prediction.notna(),
        oof.prediction.eq(oof.y_true_binarized).astype(float),
        np.nan,
    )
    oof.index.name = "ROW_ID"
    oof.to_csv(output_dir / f"oof_{STAGE_NAMES[stage]}.csv")
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(output_dir / f"folds_{STAGE_NAMES[stage]}.csv", index=False)
    return oof, fold_table


def run(stages: list[str], output_dir: Path, max_folds: int, enforce_gate: bool) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_raw = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    X, blocks = build_h4_features(X_raw)
    reference = pd.read_csv(REFERENCE_PATH, index_col="ROW_ID")
    if not reference.index.equals(X.index):
        raise ValueError("La référence V2 n'est pas alignée avec H4")
    date_sizes = X.groupby("TS", observed=True).size()
    complete_dates = date_sizes[date_sizes.eq(276)].index

    reference_metrics = stage_metrics(reference, complete_dates)
    summaries = [{"stage": "V2", **reference_metrics, "gate_pass": True}]
    uncertainty_tables = []
    executed = []
    for stage in stages:
        oof, _ = run_stage(stage, X, y, blocks, output_dir, max_folds=max_folds)
        covered = oof.prediction.notna()
        candidate = oof.loc[covered]
        ref_covered = reference.loc[covered]
        covered_dates = pd.Index(candidate.TS.unique())
        uncertainty = cluster_bootstrap_models(
            {"V2": ref_covered, STAGE_NAMES[stage]: candidate},
            covered_dates,
            reference="V2",
        )
        uncertainty["stage"] = stage
        uncertainty["scope"] = "Toutes dates"
        complete_covered = complete_dates.intersection(covered_dates)
        uncertainty_complete = cluster_bootstrap_models(
            {"V2": ref_covered, STAGE_NAMES[stage]: candidate},
            complete_covered,
            reference="V2",
        )
        uncertainty_complete["stage"] = stage
        uncertainty_complete["scope"] = "Dates complètes (276)"
        uncertainty_tables.extend([uncertainty, uncertainty_complete])

        metrics = stage_metrics(candidate, complete_dates)
        reference_local = stage_metrics(ref_covered, complete_dates)
        metrics["gain_accuracy"] = metrics["accuracy"] - reference_local["accuracy"]
        metrics["gain_complete_dates"] = metrics["accuracy_complete_dates"] - reference_local["accuracy_complete_dates"]
        metrics["gate_pass"] = metrics["gain_accuracy"] > 0 and metrics["gain_complete_dates"] > 0
        summaries.append({"stage": stage, **metrics})
        executed.append(stage)
        if enforce_gate and not metrics["gate_pass"]:
            print(f"Gate H4-{stage} non passée; arrêt séquentiel.", flush=True)
            break

    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output_dir / "results.csv", index=False)
    uncertainty_table = pd.concat(uncertainty_tables, ignore_index=True)
    uncertainty_table.to_csv(output_dir / "uncertainty.csv", index=False)
    summary = {
        "C": MODEL_C,
        "target_encoding": False,
        "date_order_used": False,
        "max_folds": max_folds,
        "enforce_gate": enforce_gate,
        "requested_stages": stages,
        "executed_stages": executed,
        "feature_blocks": blocks,
        "results": summary_table.to_dict("records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(summary_table.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=list(STAGE_NAMES), default=list(STAGE_NAMES))
    parser.add_argument("--max-folds", type=int, default=8)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "h4_nonlinear_group_features",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.stages, args.output_dir, args.max_folds, enforce_gate=not args.no_gate)

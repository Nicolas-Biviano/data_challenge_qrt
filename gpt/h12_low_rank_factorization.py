"""H12: low-rank allocation-by-state correction on top of the exact V2 logit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.h11_conditional_allocation_response import (  # noqa: E402
    ConditionalResponseDesign,
    MODEL_C,
    STATE_FEATURES,
    build_state_features,
)
from src.dataloader import ChallengeDataLoader  # noqa: E402


GPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h12_low_rank_factorization"


class LowRankStateCorrection(nn.Module):
    def __init__(self, n_allocations: int, n_states: int, rank: int, scale: float):
        super().__init__()
        self.allocation_embedding = nn.Embedding(n_allocations, rank)
        self.state_projection = nn.Linear(n_states, rank, bias=False)
        self.state_main = nn.Linear(n_states, 1, bias=False)
        self.scale = scale
        nn.init.normal_(self.allocation_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.state_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.state_main.weight)

    def forward(self, allocation: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        embedding = self.allocation_embedding(allocation)
        projected_state = self.state_projection(states)
        interaction = (embedding * projected_state).sum(dim=1)
        return self.state_main(states).squeeze(1) + self.scale * interaction


def allocation_codes(
    train: pd.Series,
    valid: pd.Series,
) -> tuple[np.ndarray, np.ndarray, int]:
    categories = pd.Index(sorted(train.astype(str).unique()))
    mapping = pd.Series(np.arange(len(categories), dtype="int64"), index=categories)
    train_code = train.astype(str).map(mapping).fillna(0).to_numpy("int64")
    valid_code = valid.astype(str).map(mapping).fillna(0).to_numpy("int64")
    return train_code, valid_code, len(categories)


def evaluate(
    model: nn.Module,
    base_logit: np.ndarray,
    states: np.ndarray,
    allocations: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        correction = model(
            torch.from_numpy(allocations),
            torch.from_numpy(states),
        ).numpy()
    logit = base_logit + correction
    accuracy = float(np.mean((logit > 0) == truth))
    loss = float(np.mean(np.logaddexp(0.0, logit) - truth * logit))
    return accuracy, loss, logit


def train_correction(
    base_train_logit: np.ndarray,
    train_states: np.ndarray,
    train_allocations: np.ndarray,
    train_truth: np.ndarray,
    base_valid_logit: np.ndarray,
    valid_states: np.ndarray,
    valid_allocations: np.ndarray,
    valid_truth: np.ndarray,
    n_allocations: int,
    rank: int,
    epochs: int,
    initialization_seed: int,
    scale: float = 0.20,
    batch_size: int = 32768,
) -> tuple[LowRankStateCorrection, pd.DataFrame, np.ndarray]:
    torch.manual_seed(initialization_seed)
    model = LowRankStateCorrection(n_allocations, len(STATE_FEATURES), rank, scale)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.state_main.parameters(), "weight_decay": 0.02},
            {
                "params": list(model.allocation_embedding.parameters())
                + list(model.state_projection.parameters()),
                "weight_decay": 0.10,
            },
        ],
        lr=0.02,
    )
    criterion = nn.BCEWithLogitsLoss()
    base_tensor = torch.from_numpy(base_train_logit.astype("float32"))
    state_tensor = torch.from_numpy(train_states)
    allocation_tensor = torch.from_numpy(train_allocations)
    truth_tensor = torch.from_numpy(train_truth.astype("float32"))
    generator = torch.Generator().manual_seed(initialization_seed)
    rng = np.random.default_rng(initialization_seed)
    train_sample = rng.choice(len(train_truth), min(100_000, len(train_truth)), replace=False)
    curves = []

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_truth), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train_truth), batch_size):
            positions = permutation[start : start + batch_size]
            correction = model(allocation_tensor[positions], state_tensor[positions])
            loss = criterion(base_tensor[positions] + correction, truth_tensor[positions])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(positions)

        train_accuracy, train_logloss, _ = evaluate(
            model,
            base_train_logit[train_sample],
            train_states[train_sample],
            train_allocations[train_sample],
            train_truth[train_sample],
        )
        valid_accuracy, valid_logloss, valid_logit = evaluate(
            model,
            base_valid_logit,
            valid_states,
            valid_allocations,
            valid_truth,
        )
        curves.append(
            {
                "epoch": epoch,
                "optimization_loss": epoch_loss / len(train_truth),
                "train_accuracy": train_accuracy,
                "valid_accuracy": valid_accuracy,
                "accuracy_gap": train_accuracy - valid_accuracy,
                "train_logloss": train_logloss,
                "valid_logloss": valid_logloss,
            }
        )
    return model, pd.DataFrame(curves), valid_logit


def run(
    output_dir: Path,
    rank: int = 4,
    epochs: int = 12,
    max_folds: int = 2,
    initialization_seed: int = 0,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = build_state_features(ChallengeDataLoader.load_X_train())
    y = ChallengeDataLoader.load_y_train()
    unique_dates = X.TS.unique()
    splits = list(KFold(n_splits=8, shuffle=True, random_state=0).split(unique_dates))
    oof_base = pd.Series(np.nan, index=X.index)
    oof_low_rank = pd.Series(np.nan, index=X.index)
    oof_fold = pd.Series(np.nan, index=X.index)
    curves = []
    fold_rows = []

    for fold, (train_date_position, valid_date_position) in enumerate(splits, 1):
        if fold > max_folds:
            break
        train_dates = unique_dates[train_date_position]
        valid_dates = unique_dates[valid_date_position]
        train_mask = X.TS.isin(train_dates)
        valid_mask = X.TS.isin(valid_dates)
        train = X.loc[train_mask]
        valid = X.loc[valid_mask]
        truth_train = y.loc[train_mask, "target_binarized"].to_numpy("int64")
        truth_valid = y.loc[valid_mask, "target_binarized"].to_numpy("int64")

        base_design = ConditionalResponseDesign([], 0.0, 0.0, 0.0).fit(train)
        train_base_matrix = base_design.transform(train)
        valid_base_matrix = base_design.transform(valid)
        base_model = LogisticRegression(
            C=MODEL_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=250,
            tol=1e-5,
            random_state=0,
        )
        base_model.fit(train_base_matrix, truth_train)
        base_train_logit = base_model.decision_function(train_base_matrix).astype("float32")
        base_valid_logit = base_model.decision_function(valid_base_matrix).astype("float32")

        state_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
        train_states = state_pipeline.fit_transform(train[STATE_FEATURES]).astype("float32")
        valid_states = state_pipeline.transform(valid[STATE_FEATURES]).astype("float32")
        train_allocations, valid_allocations, n_allocations = allocation_codes(
            train.ALLOCATION,
            valid.ALLOCATION,
        )
        _, curve, valid_logit = train_correction(
            base_train_logit,
            train_states,
            train_allocations,
            truth_train,
            base_valid_logit,
            valid_states,
            valid_allocations,
            truth_valid,
            n_allocations,
            rank,
            epochs,
            initialization_seed + fold,
        )
        curve.insert(0, "fold", fold)
        curves.append(curve)
        base_accuracy = float(np.mean((base_valid_logit > 0) == truth_valid))
        low_rank_accuracy = float(np.mean((valid_logit > 0) == truth_valid))
        fold_rows.append(
            {
                "fold": fold,
                "rank": rank,
                "epochs": epochs,
                "base_accuracy": base_accuracy,
                "low_rank_accuracy": low_rank_accuracy,
                "gain_accuracy": low_rank_accuracy - base_accuracy,
                "final_train_accuracy": float(curve.iloc[-1].train_accuracy),
                "final_valid_logloss": float(curve.iloc[-1].valid_logloss),
            }
        )
        oof_base.loc[valid_mask] = base_valid_logit
        oof_low_rank.loc[valid_mask] = valid_logit
        oof_fold.loc[valid_mask] = fold
        pd.concat(curves, ignore_index=True).to_csv(output_dir / "training_curves.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output_dir / "fold_results.csv", index=False)
        print(json.dumps(fold_rows[-1], ensure_ascii=False), flush=True)

    covered = oof_low_rank.notna()
    truth = y.loc[covered, "target_binarized"].to_numpy("int64")
    base_correct = (oof_base.loc[covered].to_numpy() > 0) == truth
    low_rank_correct = (oof_low_rank.loc[covered].to_numpy() > 0) == truth
    row_gain = low_rank_correct.astype(float) - base_correct.astype(float)
    date_gain = pd.Series(row_gain, index=X.index[covered]).groupby(X.loc[covered, "TS"]).mean()
    gain = float(row_gain.mean())
    se = float(date_gain.std(ddof=1) / np.sqrt(len(date_gain)))
    summary: dict[str, object] = {
        "model": "low-rank allocation-by-state correction over exact V2 logit",
        "rank": rank,
        "epochs_fixed": epochs,
        "initialization_seed": initialization_seed,
        "n_folds": max_folds,
        "base_accuracy": float(base_correct.mean()),
        "low_rank_accuracy": float(low_rank_correct.mean()),
        "gain_accuracy": gain,
        "date_paired_standard_error": se,
        "ci95_low": gain - 1.96 * se,
        "ci95_high": gain + 1.96 * se,
        "folds_won": int(pd.DataFrame(fold_rows).gain_accuracy.gt(0).sum()),
        "mean_final_train_valid_gap": float(
            pd.concat(curves).groupby("fold").tail(1).accuracy_gap.mean()
        ),
        "full_eight_fold_authorized": bool(
            gain > 0 and pd.DataFrame(fold_rows).gain_accuracy.gt(0).sum() >= max(1, max_folds // 2)
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    pd.DataFrame(
        {
            "fold": oof_fold.loc[covered],
            "TS": X.loc[covered, "TS"],
            "ALLOCATION": X.loc[covered, "ALLOCATION"],
            "y_true_binarized": truth,
            "base_logit": oof_base.loc[covered],
            "low_rank_logit": oof_low_rank.loc[covered],
            "base_is_correct": base_correct.astype(int),
            "low_rank_is_correct": low_rank_correct.astype(int),
        },
        index=X.index[covered],
    ).rename_axis("ROW_ID").to_csv(output_dir / "oof_predictions.csv")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=4, choices=[2, 4, 8])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            run(
                args.output_dir,
                args.rank,
                args.epochs,
                args.max_folds,
                args.initialization_seed,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

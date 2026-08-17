"""Fit the preregistered H11 allocation-response model and build a candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


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
DEFAULT_OUTPUT = GPT_DIR / "outputs" / "h11_conditional_allocation_response"
ALLOCATION_SCALE = 0.20


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_train = ChallengeDataLoader.load_X_train()
    raw_test = ChallengeDataLoader.load_X_test()
    y = ChallengeDataLoader.load_y_train()
    train = build_state_features(raw_train)
    test = build_state_features(raw_test)

    design = ConditionalResponseDesign(
        state_features=STATE_FEATURES,
        group_scale=0.0,
        allocation_scale=ALLOCATION_SCALE,
        style_scale=0.0,
    ).fit(train)
    X_train = design.transform(train)
    X_test = design.transform(test)
    model = LogisticRegression(
        C=MODEL_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=250,
        tol=1e-5,
        random_state=0,
    )
    model.fit(X_train, y.target_binarized)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability > 0.5).astype("int8")

    sample = ChallengeDataLoader.load_sample_submission()
    if not sample.index.equals(test.index):
        raise ValueError("The test and submission indices differ")
    submission = sample.copy()
    submission.iloc[:, 0] = prediction
    if submission.isna().any().any():
        raise ValueError("Missing candidate predictions")
    submission.to_csv(output_dir / "submission_h11_allocation_response.csv")
    pd.DataFrame(
        {"probability_positive": probability, "prediction": prediction},
        index=test.index,
    ).rename_axis("ROW_ID").to_csv(output_dir / "test_predictions_h11.csv")

    allocations = design.allocation_encoder.get_feature_names_out(["ALLOCATION"])
    n_allocations = len(allocations)
    n_response_coefficients = len(STATE_FEATURES) * n_allocations
    raw_response = model.coef_.ravel()[-n_response_coefficients:].reshape(
        len(STATE_FEATURES),
        n_allocations,
    )
    effective_response = raw_response * ALLOCATION_SCALE
    response_rows = []
    for state_position, state in enumerate(STATE_FEATURES):
        for allocation_position, encoded_name in enumerate(allocations):
            response_rows.append(
                {
                    "ALLOCATION": encoded_name.removeprefix("ALLOCATION_"),
                    "state": state,
                    "effective_logit_slope_per_standardized_state": float(
                        effective_response[state_position, allocation_position]
                    ),
                }
            )
    response = pd.DataFrame(response_rows)
    allocation_group = (
        raw_train.groupby("ALLOCATION", observed=True).GROUP.first().astype(int)
    )
    response["GROUP"] = response.ALLOCATION.map(allocation_group)
    response.to_csv(output_dir / "allocation_response_coefficients.csv", index=False)

    coefficient_summary = (
        response.groupby("state", observed=True)
        .effective_logit_slope_per_standardized_state.agg(
            mean="mean",
            std="std",
            minimum="min",
            q10=lambda values: values.quantile(0.10),
            median="median",
            q90=lambda values: values.quantile(0.90),
            maximum="max",
        )
        .reset_index()
        .sort_values("std", ascending=False)
    )
    coefficient_summary.to_csv(output_dir / "response_coefficient_summary.csv", index=False)
    group_response = response.pivot_table(
        index="GROUP",
        columns="state",
        values="effective_logit_slope_per_standardized_state",
        aggfunc="mean",
    )
    group_response.to_csv(output_dir / "mean_response_by_group.csv")
    response_matrix = response.pivot(
        index="ALLOCATION",
        columns="state",
        values="effective_logit_slope_per_standardized_state",
    )
    centered = response_matrix - response_matrix.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    explained = singular_values**2 / np.sum(singular_values**2)
    pd.DataFrame(
        {
            "component": np.arange(1, len(singular_values) + 1),
            "singular_value": singular_values,
            "explained_response_variance": explained,
            "cumulative_explained_response_variance": np.cumsum(explained),
        }
    ).to_csv(output_dir / "response_svd.csv", index=False)

    summary: dict[str, object] = {
        "candidate": "H11 allocation response, scale 0.20",
        "selection": "preregistered scale; not selected from the exploratory scale curve",
        "C": MODEL_C,
        "n_states": len(STATE_FEATURES),
        "n_allocations": n_allocations,
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "positive_prediction_rate": float(prediction.mean()),
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std()),
        "n_iter": int(model.n_iter_.max()),
        "submission_index_exact": bool(submission.index.equals(sample.index)),
        "response_variance_first_3_svd": float(explained[:3].sum()),
    }
    with (output_dir / "candidate_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))

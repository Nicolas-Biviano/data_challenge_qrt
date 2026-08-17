"""Diagnostic biais/variance et regularisation structuree des linear trees."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.lightgbm_v3 import standardized_matrices  # noqa: E402
from gpt.v3_features import allocation_clusters, build_v3_features  # noqa: E402
from src.dataloader import ChallengeDataLoader  # noqa: E402


CURVE_ITERATIONS = [10, 25, 50, 100, 150, 250, 400]


def model_parameters(
    config: dict[str, float | int | str],
    n_estimators: int,
) -> dict[str, object]:
    """Complete une configuration avec les parametres communs."""
    return {
        "objective": "regression_l2",
        "linear_tree": True,
        "learning_rate": 0.03,
        "n_estimators": n_estimators,
        "max_bin": 63,
        "force_col_wise": True,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 0,
        **{key: value for key, value in config.items() if key != "name"},
    }


def collect_structure(model: lgb.LGBMRegressor) -> dict[str, float | int]:
    """Resume profondeurs reelles, feuilles et coefficients locaux."""
    dump = model.booster_.dump_model()
    split_gains: list[float] = []
    leaf_counts: list[int] = []
    coefficient_norms: list[float] = []
    depths: list[int] = []
    leaves_per_tree: list[int] = []

    def visit(node: dict[str, object], depth: int, local_depths: list[int]) -> None:
        if "leaf_index" in node:
            leaf_counts.append(int(node.get("leaf_count", 0)))
            coeff = np.asarray(node.get("leaf_coeff", []), dtype=float)
            coefficient_norms.append(float(np.linalg.norm(coeff)))
            local_depths.append(depth)
            return
        split_gains.append(float(node.get("split_gain", 0.0)))
        visit(node["left_child"], depth + 1, local_depths)
        visit(node["right_child"], depth + 1, local_depths)

    for tree in dump["tree_info"]:
        local_depths: list[int] = []
        visit(tree["tree_structure"], 0, local_depths)
        depths.append(max(local_depths))
        leaves_per_tree.append(len(local_depths))

    def quantile(values: list[float] | list[int], q: float) -> float:
        return float(np.quantile(values, q)) if values else np.nan

    return {
        "n_trees": len(dump["tree_info"]),
        "actual_depth_mean": float(np.mean(depths)),
        "actual_depth_max": int(np.max(depths)),
        "leaves_per_tree_mean": float(np.mean(leaves_per_tree)),
        "leaf_count_q05": quantile(leaf_counts, 0.05),
        "leaf_count_median": quantile(leaf_counts, 0.5),
        "leaf_count_q95": quantile(leaf_counts, 0.95),
        "split_gain_q25": quantile(split_gains, 0.25),
        "split_gain_median": quantile(split_gains, 0.5),
        "split_gain_q75": quantile(split_gains, 0.75),
        "linear_coef_norm_median": quantile(coefficient_norms, 0.5),
        "linear_coef_norm_q95": quantile(coefficient_norms, 0.95),
    }


def accuracy_and_mse(
    truth_continuous: np.ndarray,
    predictions: np.ndarray,
) -> tuple[float, float]:
    accuracy = float(np.mean((predictions > 0.0) == (truth_continuous > 0.0)))
    mse = float(np.mean((predictions - truth_continuous) ** 2))
    return accuracy, mse


def build_configs(
    n_train: int,
    target_std: float,
    gain_q25: float,
    gain_median: float,
) -> list[dict[str, float | int | str]]:
    """Grille courte dont les penalites sont calibrees par taille de feuille."""
    configs: list[dict[str, float | int | str]] = []
    for depth, leaf_fraction in [(2, 0.02), (3, 0.01), (4, 0.005), (5, 0.0025)]:
        min_leaf = max(500, int(n_train * leaf_fraction))
        for strength, multiplier, gain in [
            ("moderate", 0.1, gain_q25),
            ("strong", 1.0, gain_median),
        ]:
            penalty = multiplier * min_leaf
            l1_scale = multiplier * target_std * np.sqrt(min_leaf)
            configs.append(
                {
                    "name": f"depth_{depth}_{strength}",
                    "max_depth": depth,
                    "num_leaves": min(2**depth, 24)
                    if strength == "moderate"
                    else 2 ** (depth - 1),
                    "min_child_samples": min_leaf,
                    "min_child_weight": float(min_leaf),
                    "reg_alpha": l1_scale,
                    "reg_lambda": penalty,
                    "linear_lambda": penalty,
                    "path_smooth": penalty,
                    "min_split_gain": max(0.0, gain),
                    "feature_fraction_bynode": 0.8 if strength == "moderate" else 0.6,
                    "cat_l2": 100.0 if strength == "moderate" else 500.0,
                    "cat_smooth": 100.0 if strength == "moderate" else 500.0,
                    "min_data_per_group": max(
                        100,
                        int(n_train * (0.0005 if strength == "moderate" else 0.0025)),
                    ),
                    "max_cat_threshold": 16,
                }
            )
    return configs


def fit_and_curve(
    config: dict[str, float | int | str],
    train_matrix,
    valid_matrix,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    categorical_indices: list[int],
    feature_names: list[str],
    train_eval_indices: np.ndarray,
) -> tuple[lgb.LGBMRegressor, pd.DataFrame, dict[str, float | int]]:
    """Ajuste 600 arbres et mesure la courbe sans toucher au fold externe."""
    model = lgb.LGBMRegressor(**model_parameters(config, max(CURVE_ITERATIONS)))
    model.fit(
        train_matrix,
        y_train,
        categorical_feature=categorical_indices,
        feature_name=feature_names,
    )
    rows = []
    for iteration in CURVE_ITERATIONS:
        train_prediction = model.predict(
            train_matrix[train_eval_indices],
            num_iteration=iteration,
        )
        valid_prediction = model.predict(valid_matrix, num_iteration=iteration)
        train_accuracy, train_mse = accuracy_and_mse(
            y_train[train_eval_indices],
            train_prediction,
        )
        valid_accuracy, valid_mse = accuracy_and_mse(y_valid, valid_prediction)
        rows.append(
            {
                "config": config["name"],
                "iteration": iteration,
                "train_accuracy": train_accuracy,
                "valid_accuracy": valid_accuracy,
                "accuracy_gap": train_accuracy - valid_accuracy,
                "train_mse": train_mse,
                "valid_mse": valid_mse,
            }
        )
    return model, pd.DataFrame(rows), collect_structure(model)


def run_diagnostic(outer_fold: int, output_dir: Path) -> dict[str, object]:
    """Selection interne puis evaluation unique sur le fold externe intact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ChallengeDataLoader.load_X_train()
    y = ChallengeDataLoader.load_y_train()
    features, blocks = build_v3_features(X)
    numeric_columns = blocks["returns_compact"]
    unique_ts = X["TS"].unique()
    splits = list(KFold(n_splits=8, shuffle=True, random_state=0).split(unique_ts))
    outer_train_idx, outer_valid_idx = splits[outer_fold - 1]
    outer_train_dates = unique_ts[outer_train_idx]
    outer_valid_dates = unique_ts[outer_valid_idx]
    inner_train_dates, inner_valid_dates = train_test_split(
        outer_train_dates,
        test_size=0.2,
        random_state=10_000 + outer_fold,
    )
    inner_train_mask = X["TS"].isin(inner_train_dates)
    inner_valid_mask = X["TS"].isin(inner_valid_dates)
    allocation_code = pd.Categorical(X["ALLOCATION"]).codes.astype("int32")
    group_code = pd.Categorical(X["GROUP"]).codes.astype("int32")
    cluster_code = allocation_clusters(
        X.loc[inner_train_mask],
        X["ALLOCATION"],
        n_clusters=8,
    ).to_numpy()
    train_matrix, valid_matrix, feature_names, categorical_indices = standardized_matrices(
        features,
        numeric_columns,
        inner_train_mask,
        inner_valid_mask,
        allocation_code,
        group_code,
        cluster_code,
        "full",
    )
    y_inner_train = y.loc[inner_train_mask, "target"].to_numpy()
    y_inner_valid = y.loc[inner_valid_mask, "target"].to_numpy()
    rng = np.random.default_rng(outer_fold)
    train_eval_indices = rng.choice(
        len(y_inner_train),
        size=min(100_000, len(y_inner_train)),
        replace=False,
    )

    pilot_config: dict[str, float | int | str] = {
        "name": "pilot_current",
        "max_depth": -1,
        "num_leaves": 16,
        "min_child_samples": 1000,
        "min_child_weight": 1000.0,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "linear_lambda": 10.0,
        "path_smooth": 0.0,
        "min_split_gain": 0.0,
        "feature_fraction_bynode": 1.0,
        "cat_l2": 10.0,
        "cat_smooth": 10.0,
        "min_data_per_group": 100,
        "max_cat_threshold": 32,
    }
    pilot_model, pilot_curve, pilot_structure = fit_and_curve(
        pilot_config,
        train_matrix,
        valid_matrix,
        y_inner_train,
        y_inner_valid,
        categorical_indices,
        feature_names,
        train_eval_indices,
    )
    pilot_best = pilot_curve.loc[pilot_curve["valid_accuracy"].idxmax()]
    print(
        f"pilot_current best_iter={int(pilot_best['iteration'])} "
        f"train={pilot_best['train_accuracy']:.6f} "
        f"valid={pilot_best['valid_accuracy']:.6f} "
        f"gap={pilot_best['accuracy_gap']:.6f}",
        flush=True,
    )
    del pilot_model
    gc.collect()
    configs = build_configs(
        len(y_inner_train),
        float(np.std(y_inner_train)),
        pilot_structure["split_gain_q25"],
        pilot_structure["split_gain_median"],
    )
    curves = [pilot_curve]
    structures = [{"config": "pilot_current", **pilot_structure}]
    config_lookup = {"pilot_current": pilot_config}

    def save_inner_diagnostics() -> None:
        """Conserve les diagnostics deja termines en cas d'arret couteux."""
        pd.concat(curves, ignore_index=True).to_csv(
            output_dir / f"curves_outer_{outer_fold}.csv",
            index=False,
        )
        pd.DataFrame(structures).to_csv(
            output_dir / f"structures_outer_{outer_fold}.csv",
            index=False,
        )

    save_inner_diagnostics()

    for config in configs:
        model, curve, structure = fit_and_curve(
            config,
            train_matrix,
            valid_matrix,
            y_inner_train,
            y_inner_valid,
            categorical_indices,
            feature_names,
            train_eval_indices,
        )
        del model
        gc.collect()
        curves.append(curve)
        structures.append({"config": config["name"], **structure})
        config_lookup[str(config["name"])] = config
        save_inner_diagnostics()
        best_row = curve.loc[curve["valid_accuracy"].idxmax()]
        print(
            f"{config['name']} best_iter={int(best_row['iteration'])} "
            f"train={best_row['train_accuracy']:.6f} valid={best_row['valid_accuracy']:.6f} "
            f"gap={best_row['accuracy_gap']:.6f}",
            flush=True,
        )

    curve_table = pd.concat(curves, ignore_index=True)
    structure_table = pd.DataFrame(structures)
    best_by_config = curve_table.loc[
        curve_table.groupby("config")["valid_accuracy"].idxmax()
    ].sort_values("valid_accuracy", ascending=False)
    selected_row = best_by_config.iloc[0]
    selected_name = str(selected_row["config"])
    selected_config = config_lookup[selected_name]
    selected_iterations = int(selected_row["iteration"])

    # Evaluation externe: le fold externe n'a servi ni au diagnostic ni au choix.
    outer_train_mask = X["TS"].isin(outer_train_dates)
    outer_valid_mask = X["TS"].isin(outer_valid_dates)
    outer_cluster = allocation_clusters(
        X.loc[outer_train_mask],
        X["ALLOCATION"],
        n_clusters=8,
    ).to_numpy()
    outer_train_matrix, outer_valid_matrix, feature_names, categorical_indices = (
        standardized_matrices(
            features,
            numeric_columns,
            outer_train_mask,
            outer_valid_mask,
            allocation_code,
            group_code,
            outer_cluster,
            "full",
        )
    )
    outer_model = lgb.LGBMRegressor(
        **model_parameters(selected_config, selected_iterations)
    )
    outer_model.fit(
        outer_train_matrix,
        y.loc[outer_train_mask, "target"],
        categorical_feature=categorical_indices,
        feature_name=feature_names,
    )
    outer_prediction = outer_model.predict(outer_valid_matrix)
    outer_accuracy, outer_mse = accuracy_and_mse(
        y.loc[outer_valid_mask, "target"].to_numpy(),
        outer_prediction,
    )
    external = {
        "outer_fold": outer_fold,
        "selected_config": selected_name,
        "selected_iterations": selected_iterations,
        "inner_valid_accuracy": float(selected_row["valid_accuracy"]),
        "inner_train_accuracy": float(selected_row["train_accuracy"]),
        "outer_accuracy": outer_accuracy,
        "outer_mse": outer_mse,
        "outer_n_train": int(outer_train_mask.sum()),
        "outer_n_valid": int(outer_valid_mask.sum()),
    }
    curve_table.to_csv(output_dir / f"curves_outer_{outer_fold}.csv", index=False)
    structure_table.to_csv(output_dir / f"structures_outer_{outer_fold}.csv", index=False)
    best_by_config.to_csv(output_dir / f"best_by_config_outer_{outer_fold}.csv", index=False)
    pd.DataFrame(
        {
            "ROW_ID": X.index[outer_valid_mask],
            "prediction_score": outer_prediction,
            "target": y.loc[outer_valid_mask, "target"].to_numpy(),
        }
    ).to_csv(output_dir / f"outer_predictions_{outer_fold}.csv", index=False)
    with (output_dir / f"selection_outer_{outer_fold}.json").open("w") as stream:
        json.dump(
            {
                "external": external,
                "selected_parameters": model_parameters(
                    selected_config,
                    selected_iterations,
                ),
                "pilot_structure": pilot_structure,
            },
            stream,
            indent=2,
        )
    print(json.dumps(external, indent=2), flush=True)
    return external


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-fold", type=int, choices=range(1, 9), default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "v3_lgbm_diagnostics",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnostic(args.outer_fold, args.output_dir)

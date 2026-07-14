"""Analysis for the fixed sparse-montage feasibility screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .method_compare import paired_method_bootstrap


def _load_aligned_views(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    path = Path(run_dir) / "predictions.csv"
    frame = pd.read_csv(path)
    required = {"test_view", "trial_index", "subject_id", "y_true", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    views = {
        str(view).casefold(): group.sort_values(["subject_id", "trial_index"]).reset_index(drop=True)
        for view, group in frame.groupby("test_view", sort=False)
    }
    if "car" not in views:
        raise ValueError(f"{path} must contain a CAR view")
    car = views["car"]
    identifiers = ["subject_id", "trial_index", "y_true"]
    for view, current in views.items():
        if not np.array_equal(car[identifiers].to_numpy(), current[identifiers].to_numpy()):
            raise RuntimeError(f"Prediction trials for {view} are not aligned in {path}")
    return car, views


def _class_recall_gaps(car: pd.DataFrame, shifted: pd.DataFrame) -> np.ndarray:
    y_true = car["y_true"].to_numpy(dtype=np.int64)
    car_prediction = car["y_pred"].to_numpy(dtype=np.int64)
    shifted_prediction = shifted["y_pred"].to_numpy(dtype=np.int64)
    classes = np.unique(y_true)
    if not np.array_equal(classes, np.arange(classes.size)):
        raise ValueError(f"Expected contiguous class indices, found {classes.tolist()}")
    gaps = np.empty(classes.size, dtype=np.float64)
    for class_index in classes:
        mask = y_true == class_index
        gaps[class_index] = np.mean(car_prediction[mask] == class_index) - np.mean(
            shifted_prediction[mask] == class_index
        )
    return gaps


def analyze_montage_screen(
    car_only_dir: str | Path,
    canonical_dir: str | Path,
    augmentation_dir: str | Path,
    consistency_dir: str | Path,
    output_dir: str | Path,
    *,
    primary_view: str = "sparse16@cz",
    target_class: int = 0,
    selected_lambda: float = 10.0,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20260714,
) -> pd.DataFrame:
    """Summarize all views and bootstrap the predeclared primary comparison."""

    named_runs = {
        "car_only": car_only_dir,
        "car_canonicalize": canonical_dir,
        "multi_view_ce": augmentation_dir,
        "rule_consistency": consistency_dir,
    }
    rows: list[dict[str, object]] = []
    expected_views: list[str] | None = None
    expected_ids: np.ndarray | None = None
    identifiers = ["subject_id", "trial_index", "y_true"]

    for method, run_dir in named_runs.items():
        metrics = pd.read_csv(Path(run_dir) / "metrics.csv")
        car, views = _load_aligned_views(run_dir)
        current_views = [str(view).casefold() for view in metrics["test_view"].tolist()]
        if expected_views is None:
            expected_views = current_views
            expected_ids = car[identifiers].to_numpy()
        elif current_views != expected_views:
            raise RuntimeError(f"Observation views differ for method {method}: {current_views}")
        if expected_ids is not None and not np.array_equal(expected_ids, car[identifiers].to_numpy()):
            raise RuntimeError(f"Prediction trials differ for method {method}")

        car_metric = metrics[metrics["test_view"].str.casefold() == "car"].iloc[0]
        for _, metric in metrics.iterrows():
            view = str(metric["test_view"])
            shifted = views[view.casefold()]
            gaps = _class_recall_gaps(car, shifted)
            worst_class = int(np.argmax(gaps))
            row = {
                "method": method,
                "run_dir": str(run_dir),
                "test_view": view,
                "reference_view": metric.get("reference_view", view),
                "montage": metric.get("montage", "full"),
                "n_observed_channels": metric.get("n_observed_channels", np.nan),
                "balanced_accuracy": float(metric["balanced_accuracy"]),
                "balanced_accuracy_gap_from_car": float(
                    car_metric["balanced_accuracy"] - metric["balanced_accuracy"]
                ),
                "prediction_disagreement_from_car": float(
                    np.mean(car["y_pred"].to_numpy() != shifted["y_pred"].to_numpy())
                ),
                "target_class_recall_gap": float(gaps[target_class]),
                "worst_class_recall_gap": float(gaps[worst_class]),
                "worst_class_index": worst_class,
            }
            rows.append(row)

    result = pd.DataFrame(rows)
    baseline = result[result["method"] == "car_only"].set_index("test_view")
    for index, row in result.iterrows():
        base_row = baseline.loc[row["test_view"]]
        result.loc[index, "balanced_accuracy_gain_vs_car_only"] = (
            float(row["balanced_accuracy"]) - float(base_row["balanced_accuracy"])
        )
        result.loc[index, "target_class_gap_recovery_vs_car_only"] = (
            float(base_row["target_class_recall_gap"]) - float(row["target_class_recall_gap"])
        )

    primary_key = primary_view.casefold()
    primary_rows = result[result["test_view"].str.casefold() == primary_key].set_index("method")
    if len(primary_rows) != len(named_runs):
        raise ValueError(f"Primary view {primary_view!r} is missing from one or more methods")
    paired = paired_method_bootstrap(
        augmentation_dir,
        consistency_dir,
        target_view=primary_view,
        target_class=target_class,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=bootstrap_seed,
    )
    primary_bacc = paired[paired["metric"] == "target_balanced_accuracy_gain"].iloc[0]
    car_only_gap = float(primary_rows.loc["car_only", "balanced_accuracy_gap_from_car"])
    hard_shift = car_only_gap >= 0.03
    summary = {
        "stage": "E7a sparse-montage feasibility screen",
        "probe_seeds": [int(pd.read_csv(Path(car_only_dir) / "metrics.csv")["probe_seed"].iloc[0])],
        "primary_view": primary_view,
        "target_class_index": int(target_class),
        "selected_consistency_lambda": float(selected_lambda),
        "lambda_was_selected_before_montage_targets": True,
        "missing_channel_policy": "zero-fill after physical re-referencing; preserve channel order",
        "car_canonicalization_is_exact_for_sparse_views": False,
        "hard_shift_threshold": 0.03,
        "hard_shift_detected_on_primary": bool(hard_shift),
        "primary_car_only_balanced_accuracy_gap": car_only_gap,
        "primary_rule_vs_augmentation_target_bacc": {
            key: primary_bacc[key].item() if isinstance(primary_bacc[key], np.generic) else primary_bacc[key]
            for key in (
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "probability_improvement_positive",
            )
        },
        "next_decision": (
            "Proceed to montage-aware training; this screen is one-seed exploratory evidence only."
            if hard_shift
            else "Do not train a new method yet; strengthen or revise the sparse-montage stressor."
        ),
        "interpretation_rule": (
            "Do not tune lambda or select a different primary montage from these test rows. "
            "Use this screen to decide whether missing channels expose a limitation; any new "
            "montage-aware method requires validation-only selection and a later three-seed audit."
        ),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path / "montage_method_by_view.csv", index=False)
    paired.to_csv(output_path / "primary_paired_method_bootstrap.csv", index=False)
    with (output_path / "montage_screen_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return result

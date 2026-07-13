"""Predeclared comparison for the held-out-reference consistency screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _run_effect(run_dir: str | Path, *, target_view: str, target_class: int) -> dict[str, float]:
    run_path = Path(run_dir)
    metrics = pd.read_csv(run_path / "metrics.csv")
    predictions = pd.read_csv(run_path / "predictions.csv")
    car_metric = metrics[metrics["test_view"].str.casefold() == "car"].iloc[0]
    shifted_metric = metrics[metrics["test_view"].str.casefold() == target_view.casefold()].iloc[0]

    car = predictions[predictions["test_view"].str.casefold() == "car"].sort_values("trial_index")
    shifted = predictions[
        predictions["test_view"].str.casefold() == target_view.casefold()
    ].sort_values("trial_index")
    if not np.array_equal(car["y_true"].to_numpy(), shifted["y_true"].to_numpy()):
        raise RuntimeError(f"Prediction labels are not aligned in {run_path}")
    y_true = car["y_true"].to_numpy()
    class_mask = y_true == target_class
    recall_gap = float(
        np.mean(car.loc[class_mask, "y_pred"].to_numpy() == target_class)
        - np.mean(shifted.loc[class_mask, "y_pred"].to_numpy() == target_class)
    )
    return {
        "car_balanced_accuracy": float(car_metric["balanced_accuracy"]),
        "target_balanced_accuracy": float(shifted_metric["balanced_accuracy"]),
        "balanced_accuracy_gap": float(car_metric["balanced_accuracy"] - shifted_metric["balanced_accuracy"]),
        "target_class_recall_gap": recall_gap,
    }


def compare_consistency_methods(
    baseline_dir: str | Path,
    augmentation_dir: str | Path,
    consistency_dir: str | Path,
    output_dir: str | Path,
    *,
    target_view: str = "cz",
    target_class: int = 0,
) -> pd.DataFrame:
    named_runs = {
        "car_only": baseline_dir,
        "multi_view_ce": augmentation_dir,
        "rule_consistency": consistency_dir,
    }
    rows = []
    for method, run_dir in named_runs.items():
        effect = _run_effect(run_dir, target_view=target_view, target_class=target_class)
        rows.append({"method": method, "run_dir": str(run_dir), **effect})
    comparison = pd.DataFrame(rows)
    baseline = comparison[comparison["method"] == "car_only"].iloc[0]
    for column in ("balanced_accuracy_gap", "target_class_recall_gap"):
        baseline_gap = float(baseline[column])
        comparison[f"{column}_relative_reduction"] = (
            baseline_gap - comparison[column]
        ) / max(abs(baseline_gap), 1e-12)
    comparison["clean_balanced_accuracy_change"] = (
        comparison["car_balanced_accuracy"] - float(baseline["car_balanced_accuracy"])
    )

    augmentation = comparison[comparison["method"] == "multi_view_ce"].iloc[0]
    consistency = comparison[comparison["method"] == "rule_consistency"].iloc[0]
    summary = {
        "target_view": target_view,
        "target_class_index": int(target_class),
        "predeclared_primary_metric": "held-out-view target-class recall gap",
        "success_thresholds": {
            "relative_recall_gap_reduction_vs_car_only": 0.30,
            "maximum_clean_balanced_accuracy_drop": 0.01,
        },
        "augmentation_passes": bool(
            augmentation["target_class_recall_gap_relative_reduction"] >= 0.30
            and augmentation["clean_balanced_accuracy_change"] >= -0.01
        ),
        "consistency_passes": bool(
            consistency["target_class_recall_gap_relative_reduction"] >= 0.30
            and consistency["clean_balanced_accuracy_change"] >= -0.01
        ),
        "consistency_beats_augmentation_on_primary": bool(
            consistency["target_class_recall_gap"] < augmentation["target_class_recall_gap"]
        ),
        "interpretation_rule": (
            "Claim rule-loss value only if rule_consistency passes and beats multi_view_ce; "
            "otherwise attribute recovery to augmentation."
        ),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path / "method_comparison.csv", index=False)
    with (output_path / "method_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return comparison

"""Validity checks for REVE's native variable-channel montage screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import paired_subject_bootstrap_bacc_gap


def _prediction_views(run_dir: str | Path) -> dict[str, pd.DataFrame]:
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
        raise ValueError(f"{path} must contain CAR predictions")
    identifiers = ["subject_id", "trial_index", "y_true"]
    for view, current in views.items():
        if not np.array_equal(views["car"][identifiers].to_numpy(), current[identifiers].to_numpy()):
            raise RuntimeError(f"Prediction trials for {view} are not aligned in {path}")
    return views


def _prediction_distribution(frame: pd.DataFrame, n_classes: int) -> dict[str, float | int]:
    counts = np.bincount(frame["y_pred"].to_numpy(dtype=np.int64), minlength=n_classes)
    probability = counts / counts.sum()
    positive = probability[probability > 0.0]
    entropy = float(-(positive * np.log(positive)).sum() / np.log(n_classes))
    return {
        "n_predicted_classes": int(np.sum(counts > 0)),
        "prediction_entropy_normalized": entropy,
        "largest_predicted_class_fraction": float(probability.max()),
    }


def _method_rows(method: str, run_dir: str | Path) -> pd.DataFrame:
    metrics = pd.read_csv(Path(run_dir) / "metrics.csv")
    views = _prediction_views(run_dir)
    n_classes = int(views["car"]["y_true"].nunique())
    car_bacc = float(metrics.loc[metrics["test_view"].str.casefold() == "car", "balanced_accuracy"].iloc[0])
    rows = []
    for _, metric in metrics.iterrows():
        view = str(metric["test_view"])
        rows.append(
            {
                "method": method,
                "run_dir": str(run_dir),
                "probe": metric.get("probe", "unknown"),
                "set_queries": int(metric.get("set_queries", 0)),
                "test_view": view,
                "reference_view": metric.get("reference_view", view),
                "montage": metric.get("montage", "full"),
                "n_observed_channels": metric.get("n_observed_channels", np.nan),
                "balanced_accuracy": float(metric["balanced_accuracy"]),
                "balanced_accuracy_gap_from_full_car": car_bacc - float(metric["balanced_accuracy"]),
                "macro_f1": float(metric["macro_f1"]),
                "macro_auroc_ovr": float(metric["macro_auroc_ovr"]),
                "paired_cosine_to_full_car": float(metric["paired_cosine_to_car"]),
                "linear_cka_to_full_car": float(metric["linear_cka_to_car"]),
                **_prediction_distribution(views[view.casefold()], n_classes),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_primary(
    run_dir: str | Path,
    primary_view: str,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    views = _prediction_views(run_dir)
    car = views["car"]
    target = views[primary_view.casefold()]
    return paired_subject_bootstrap_bacc_gap(
        car["y_true"].to_numpy(dtype=np.int64),
        car["y_pred"].to_numpy(dtype=np.int64),
        target["y_pred"].to_numpy(dtype=np.int64),
        car["subject_id"].to_numpy(dtype=np.int64),
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


def analyze_native_montage_screen(
    baseline_dir: str | Path,
    canonical_dir: str | Path,
    output_dir: str | Path,
    *,
    primary_view: str = "native16@cz",
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20260714,
) -> pd.DataFrame:
    """Decide whether native channel removal yields a usable harder benchmark."""

    result = pd.concat(
        [
            _method_rows("full_car_probe", baseline_dir),
            _method_rows("car_canonicalize", canonical_dir),
        ],
        ignore_index=True,
    )
    baseline = result[result["method"] == "full_car_probe"].set_index("test_view")
    canonical = result[result["method"] == "car_canonicalize"].set_index("test_view")
    required_views = {
        "car",
        "native32@car", "native16@car", "native8@car",
        "native32@cz", "native16@cz", "native8@cz",
    }
    missing = sorted(required_views - set(baseline.index.str.casefold()))
    if missing:
        raise ValueError(f"Native montage screen is missing views: {missing}")

    primary = baseline.loc[primary_view]
    clean = baseline.loc["car"]
    chance = 1.0 / 4.0
    hard_shift = float(primary["balanced_accuracy_gap_from_full_car"]) >= 0.03
    noncollapse = bool(
        int(primary["n_predicted_classes"]) >= 3
        and float(primary["prediction_entropy_normalized"]) >= 0.30
        and float(primary["largest_predicted_class_fraction"]) <= 0.95
        and float(primary["balanced_accuracy"]) >= chance + 0.02
    )
    clean_gate = float(clean["balanced_accuracy"]) >= 0.45
    car_curve = [float(baseline.loc[f"native{size}@car", "balanced_accuracy"]) for size in (32, 16, 8)]
    monotone = bool(car_curve[0] >= car_curve[1] >= car_curve[2])
    if not clean_gate:
        status = "clean_gate_failed"
    elif not noncollapse:
        status = "invalid_prediction_collapse"
    elif not hard_shift:
        status = "valid_but_insufficient_stress"
    else:
        status = "usable_native_montage_benchmark"

    primary_bootstrap = _bootstrap_primary(
        baseline_dir,
        primary_view,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=bootstrap_seed,
    )
    primary_size = primary_view.split("@", maxsplit=1)[0]
    paired_car_view = f"{primary_size}@car"
    probe_name = str(clean["probe"])
    set_queries = int(clean["set_queries"])
    if probe_name.casefold() == "reve_set":
        stage = "E7c variable-set native channel-subset validity screen"
        encoder_readout = f"frozen REVE tokens plus validation-selected {set_queries}-query set probe"
    else:
        stage = "E7b native channel-subset validity screen"
        encoder_readout = "REVE released attention pooling plus sklearn linear probe"
    summary = {
        "stage": stage,
        "primary_view": primary_view,
        "channel_policy": (
            "select retained electrodes, reference within that montage, and remove other signals/coordinates"
        ),
        "encoder_readout": encoder_readout,
        "probe": probe_name,
        "set_queries": set_queries,
        "clean_gate_passed": clean_gate,
        "clean_car_balanced_accuracy": float(clean["balanced_accuracy"]),
        "chance_balanced_accuracy": chance,
        "hard_shift_threshold": 0.03,
        "hard_shift_detected": hard_shift,
        "primary_balanced_accuracy": float(primary["balanced_accuracy"]),
        "primary_gap_from_full_car": float(primary["balanced_accuracy_gap_from_full_car"]),
        "primary_prediction_noncollapse_passed": noncollapse,
        "primary_prediction_entropy_normalized": float(primary["prediction_entropy_normalized"]),
        "primary_largest_predicted_class_fraction": float(primary["largest_predicted_class_fraction"]),
        "primary_subject_bootstrap": primary_bootstrap,
        "montage_only_drop": float(
            clean["balanced_accuracy"] - baseline.loc[paired_car_view, "balanced_accuracy"]
        ),
        "additional_reference_drop_within_montage": float(
            baseline.loc[paired_car_view, "balanced_accuracy"] - primary["balanced_accuracy"]
        ),
        "canonical_reference_residual_within_montage": float(
            canonical.loc[paired_car_view, "balanced_accuracy"]
            - canonical.loc[primary_view, "balanced_accuracy"]
        ),
        "native_car_curve_32_16_8": car_curve,
        "native_car_monotonicity_passed": monotone,
        "benchmark_status": status,
        "decision_rule": (
            "Proceed to montage-aware learning only when the clean gate passes, the primary full-to-native "
            "gap is at least 0.03, and native predictions avoid the zero-fill collapse signature."
        ),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path / "native_montage_by_view.csv", index=False)
    with (output_path / "native_montage_screen_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return result

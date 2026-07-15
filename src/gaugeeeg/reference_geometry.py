"""E7e geometry-controlled native-reference diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .reference_closure import (
    _bootstrap_recall_gap,
    _class_auroc,
    _load_views,
    _recall,
)


def analyze_reference_geometry(
    run_dir: str | Path,
    e7d_full_run_dir: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20260715,
) -> pd.DataFrame:
    """Measure native CAR/Cz/Pz/Fz shifts without selecting a test reference."""
    views, probability_columns = _load_views(run_dir)
    full_views, full_probability_columns = _load_views(e7d_full_run_dir)
    references = ("cz", "pz", "fz")
    montage_sizes = (32, 16)
    required = {"car"}
    for size in montage_sizes:
        required.add(f"native{size}@car")
        required.update(f"native{size}@{reference}" for reference in references)
    if missing := sorted(required - set(views)):
        raise ValueError(f"Reference-geometry run is missing views: {missing}")
    if probability_columns != full_probability_columns or not probability_columns:
        raise ValueError("E7d and E7e require the same complete probability columns")

    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    selected_queries = int(selection["selected_queries"])
    metrics = pd.read_csv(Path(run_dir) / "metrics.csv")
    if set(metrics["set_queries"].astype(int)) != {selected_queries}:
        raise ValueError("Reference-geometry run does not use the frozen query count")
    metrics_by_view = {
        str(row["test_view"]).casefold(): row for _, row in metrics.iterrows()
    }

    identifiers = ["subject_id", "trial_index", "y_true"]
    car = views["car"]
    e7d_car = full_views["car"]
    car_trials_match = bool(
        np.array_equal(car[identifiers].to_numpy(), e7d_car[identifiers].to_numpy())
    )
    car_predictions_match = bool(
        car_trials_match
        and np.array_equal(car["y_pred"].to_numpy(), e7d_car["y_pred"].to_numpy())
    )
    if not car_predictions_match:
        raise RuntimeError("E7e did not reproduce the deterministic E7d q4 CAR predictions")

    class_names = [column.removeprefix("prob_") for column in probability_columns]
    view_rows: list[dict[str, float | int | str]] = []
    class_rows: list[dict[str, float | int | str]] = []
    bootstrap_rows: list[dict[str, float | int | str | bool]] = []
    for size_index, size in enumerate(montage_sizes):
        base_view = f"native{size}@car"
        base = views[base_view]
        y_true = base["y_true"].to_numpy(dtype=np.int64)
        base_prediction = base["y_pred"].to_numpy(dtype=np.int64)
        base_bacc = float(balanced_accuracy_score(y_true, base_prediction))
        for reference_index, reference in enumerate(references):
            target_view = f"native{size}@{reference}"
            target = views[target_view]
            target_prediction = target["y_pred"].to_numpy(dtype=np.int64)
            target_bacc = float(balanced_accuracy_score(y_true, target_prediction))
            metric = metrics_by_view[target_view]
            view_rows.append(
                {
                    "montage_size": size,
                    "base_view": base_view,
                    "target_view": target_view,
                    "reference": reference,
                    "base_balanced_accuracy": base_bacc,
                    "target_balanced_accuracy": target_bacc,
                    "balanced_accuracy_gap": base_bacc - target_bacc,
                    "prediction_disagreement": float(
                        np.mean(base_prediction != target_prediction)
                    ),
                    "target_macro_auroc_ovr": float(metric["macro_auroc_ovr"]),
                    "paired_cosine_to_full_car": float(metric["paired_cosine_to_car"]),
                    "linear_cka_to_full_car": float(metric["linear_cka_to_car"]),
                }
            )
            pair_offset = size_index * len(references) + reference_index
            for class_index, (class_name, probability_column) in enumerate(
                zip(class_names, probability_columns, strict=True)
            ):
                base_recall = _recall(base, class_index)
                target_recall = _recall(target, class_index)
                base_auroc = _class_auroc(base, probability_column, class_index)
                target_auroc = _class_auroc(target, probability_column, class_index)
                class_rows.append(
                    {
                        "montage_size": size,
                        "base_view": base_view,
                        "target_view": target_view,
                        "reference": reference,
                        "class_index": class_index,
                        "class_name": class_name,
                        "base_recall": base_recall,
                        "target_recall": target_recall,
                        "recall_gap": base_recall - target_recall,
                        "base_predicted_fraction": float(
                            np.mean(base_prediction == class_index)
                        ),
                        "target_predicted_fraction": float(
                            np.mean(target_prediction == class_index)
                        ),
                        "base_auroc": base_auroc,
                        "target_auroc": target_auroc,
                        "auroc_gap": base_auroc - target_auroc,
                    }
                )
                bootstrap = _bootstrap_recall_gap(
                    base,
                    target,
                    class_index,
                    n_resamples=n_resamples,
                    confidence=confidence,
                    seed=bootstrap_seed + pair_offset * 100 + class_index,
                )
                bootstrap_rows.append(
                    {
                        "montage_size": size,
                        "base_view": base_view,
                        "target_view": target_view,
                        "reference": reference,
                        "class_index": class_index,
                        "class_name": class_name,
                        "point_recall_gap": base_recall - target_recall,
                        **bootstrap,
                        "ci_excludes_zero": bool(
                            float(bootstrap["ci_lower"]) > 0.0
                            or float(bootstrap["ci_upper"]) < 0.0
                        ),
                    }
                )

    view_frame = pd.DataFrame(view_rows)
    class_frame = pd.DataFrame(class_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    native16_views = view_frame[view_frame["montage_size"] == 16]
    alternative_native16 = native16_views[native16_views["reference"].isin(["pz", "fz"])]
    alternative_classes = class_frame[
        (class_frame["montage_size"] == 16) & class_frame["reference"].isin(["pz", "fz"])
    ]
    alternative_bootstrap = bootstrap_frame[
        (bootstrap_frame["montage_size"] == 16)
        & bootstrap_frame["reference"].isin(["pz", "fz"])
    ]
    aggregate_shift_supported = bool(
        (alternative_native16["balanced_accuracy_gap"].abs() >= 0.03).any()
    )
    class_candidates = alternative_classes[alternative_classes["recall_gap"].abs() >= 0.10]
    class_shift_supported = False
    for _, candidate in class_candidates.iterrows():
        interval = alternative_bootstrap[
            (alternative_bootstrap["target_view"] == candidate["target_view"])
            & (alternative_bootstrap["class_index"] == candidate["class_index"])
        ].iloc[0]
        if bool(interval["ci_excludes_zero"]):
            class_shift_supported = True
            break
    disagreement_supported = bool(
        (alternative_native16["prediction_disagreement"] >= 0.15).any()
    )
    joint_target_supported = bool(aggregate_shift_supported or class_shift_supported)
    largest_view = native16_views.iloc[
        native16_views["balanced_accuracy_gap"].abs().argmax()
    ]
    largest_class = alternative_classes.iloc[
        alternative_classes["recall_gap"].abs().argmax()
    ]

    e7d_metrics = pd.read_csv(Path(e7d_full_run_dir) / "metrics.csv")
    e7d_pz = e7d_metrics[e7d_metrics["test_view"].astype(str).str.casefold() == "pz"].iloc[0]
    summary = {
        "stage": "E7e geometry-controlled native-reference diagnostic",
        "selected_queries_frozen": selected_queries,
        "predeclared_references": list(references),
        "predeclared_montage_sizes": list(montage_sizes),
        "selection_rule": "No reference is selected from E7e test; report the complete suite.",
        "car_trials_reproduced": car_trials_match,
        "car_predictions_reproduced_exactly": car_predictions_match,
        "e7d_full_pz_balanced_accuracy_gap_for_context": float(
            e7d_pz["balanced_accuracy_gap_from_car"]
        ),
        "native16_by_reference": [
            {
                "reference": str(row["reference"]),
                "balanced_accuracy_gap": float(row["balanced_accuracy_gap"]),
                "prediction_disagreement": float(row["prediction_disagreement"]),
            }
            for _, row in native16_views.iterrows()
        ],
        "native16_largest_absolute_bacc_gap_reference": str(largest_view["reference"]),
        "native16_largest_absolute_bacc_gap": float(abs(largest_view["balanced_accuracy_gap"])),
        "native16_alternative_largest_absolute_class_recall_gap_reference": str(
            largest_class["reference"]
        ),
        "native16_alternative_largest_absolute_class_recall_gap_class": str(
            largest_class["class_name"]
        ),
        "native16_alternative_largest_absolute_class_recall_gap": float(
            abs(largest_class["recall_gap"])
        ),
        "native16_alternative_aggregate_shift_supported": aggregate_shift_supported,
        "native16_alternative_class_shift_supported": class_shift_supported,
        "native16_alternative_disagreement_supported": disagreement_supported,
        "joint_gauge_montage_target_supported": joint_target_supported,
        "method_scope_decision": (
            "joint_gauge_montage_learning"
            if joint_target_supported
            else "montage_primary_with_gauge_auxiliary"
        ),
        "decision_rule": (
            "Joint scope is supported when native16 Pz/Fz produces either an absolute BAcc gap "
            ">=0.03 or an absolute class-recall gap >=0.10 with a subject-bootstrap CI excluding zero."
        ),
        "geometry_caveat": (
            "Cz/Pz/Fz are named-electrode diagnostics, not a calibrated continuous measure of "
            "distance from the montage centroid."
        ),
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    view_frame.to_csv(destination / "reference_geometry_by_view.csv", index=False)
    class_frame.to_csv(destination / "reference_geometry_class_metrics.csv", index=False)
    bootstrap_frame.to_csv(destination / "reference_geometry_class_bootstrap.csv", index=False)
    with (destination / "reference_geometry_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return view_frame

"""Locked development audit for the Gauge-Selective Residual Adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_benchmark import (
    EVALUATION_VIEWS,
    JOINT_TRAINING_VIEWS,
    BaselineSpec,
    _hierarchical_bacc_delta_views,
    _metric_row,
    _validate_run,
)

PRIMARY_VIEWS = tuple(view for view in EVALUATION_VIEWS if view.startswith("native16@"))
NATIVE32_VIEWS = tuple(view for view in EVALUATION_VIEWS if view.startswith("native32@"))


@dataclass(frozen=True)
class GSRASpec:
    auxiliary: str
    fusion: str
    preservation_weight: float
    residual_consistency_weight: float
    gate_supervision_weight: float


GSRA_SPECS: dict[str, GSRASpec] = {
    "joint_multiview_ce": GSRASpec("none", "residual", 0.0, 0.0, 0.0),
    "gqba_odd_even": GSRASpec("gqba_odd_even", "residual", 0.0, 0.0, 0.0),
    "gated_spectral_control": GSRASpec("spectral_capacity_control", "gated_residual", 0.0, 0.0, 0.0),
    "gqba_gated_ce": GSRASpec("gqba_odd_even", "gated_residual", 0.0, 0.0, 0.0),
    "gsra": GSRASpec("gqba_odd_even", "gated_residual", 1.0, 0.1, 0.1),
}
NEW_METHODS = ("gated_spectral_control", "gqba_gated_ce", "gsra")


def _parse_specs(run_specs: list[str]) -> dict[str, list[Path]]:
    grouped = {method: [] for method in GSRA_SPECS}
    for item in run_specs:
        if "=" not in item:
            raise ValueError(f"Invalid run specification {item!r}; expected method=path")
        method, path = item.split("=", maxsplit=1)
        if method not in grouped:
            raise ValueError(f"Unknown GSRA method {method!r}; expected {sorted(grouped)}")
        grouped[method].append(Path(path))
    missing = [method for method, paths in grouped.items() if not paths]
    if missing:
        raise ValueError(f"Missing required GSRA runs: {missing}")
    return grouped


def _class_recall(frame: pd.DataFrame, class_index: int) -> float:
    labels = frame["y_true"].to_numpy(dtype=np.int64)
    predictions = frame["y_pred"].to_numpy(dtype=np.int64)
    mask = labels == class_index
    if not np.any(mask):
        raise ValueError(f"Class {class_index} is absent from a bootstrap sample")
    return float(np.mean(predictions[mask] == class_index))


def _hierarchical_recall_delta_views(
    candidate: dict[int, dict[str, pd.DataFrame]],
    baseline: dict[int, dict[str, pd.DataFrame]],
    *,
    views: tuple[str, ...],
    class_index: int,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    probe_seeds = sorted(candidate)
    if probe_seeds != sorted(baseline):
        raise ValueError("Candidate and baseline probe seeds differ")
    keys = ["trial_index", "subject_id", "y_true"]
    point_values = []
    for probe_seed in probe_seeds:
        view_values = []
        for view in views:
            candidate_frame = candidate[probe_seed][view]
            baseline_frame = baseline[probe_seed][view]
            if not candidate_frame[keys].equals(baseline_frame[keys]):
                raise ValueError("Candidate and baseline predictions are not paired")
            view_values.append(
                _class_recall(candidate_frame, class_index) - _class_recall(baseline_frame, class_index)
            )
        point_values.append(float(np.mean(view_values)))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for draw_index in range(n_resamples):
        seed_values = []
        for sampled_seed in rng.choice(probe_seeds, size=len(probe_seeds), replace=True):
            probe_seed = int(sampled_seed)
            first = candidate[probe_seed][views[0]]
            subjects = first["subject_id"].unique()
            sampled_subjects = rng.choice(subjects, size=subjects.size, replace=True)
            positions = np.concatenate(
                [np.flatnonzero(first["subject_id"].to_numpy() == subject) for subject in sampled_subjects]
            )
            view_values = []
            for view in views:
                candidate_frame = candidate[probe_seed][view].iloc[positions]
                baseline_frame = baseline[probe_seed][view].iloc[positions]
                view_values.append(
                    _class_recall(candidate_frame, class_index) - _class_recall(baseline_frame, class_index)
                )
            seed_values.append(float(np.mean(view_values)))
        draws[draw_index] = float(np.mean(seed_values))
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_probe_seeds": len(probe_seeds),
        "n_resamples": n_resamples,
        "confidence": confidence,
        "candidate_minus_baseline": float(np.mean(point_values)),
        "bootstrap_mean": float(draws.mean()),
        "ci_lower": float(np.quantile(draws, alpha)),
        "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
    }


def analyze_gsra_benchmark(
    run_specs: list[str],
    output_dir: str | Path,
    *,
    expected_seeds: list[int] | tuple[int, ...] = (7, 21, 42),
    bootstrap_resamples: int = 10_000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260723,
) -> pd.DataFrame:
    """Validate the capacity-matched GSRA screen and write its frozen decision."""

    grouped = _parse_specs(run_specs)
    expected = {int(seed) for seed in expected_seeds}
    frames: dict[str, dict[int, dict[str, pd.DataFrame]]] = {}
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    new_parameter_counts: set[tuple[int, int]] = set()
    fingerprints: set[str] = set()
    revisions: set[tuple[str, str]] = set()

    common_spec = BaselineSpec(JOINT_TRAINING_VIEWS, "multi_view_ce")
    for method, paths in grouped.items():
        frames[method] = {}
        observed: set[int] = set()
        expected_spec = GSRA_SPECS[method]
        for path in paths:
            summary, by_view = _validate_run(method, path, spec=common_spec)
            probe_seed = int(summary["probe_seed"])
            if probe_seed in observed:
                raise ValueError(f"Duplicate seed {probe_seed} for {method}")
            observed.add(probe_seed)
            auxiliary = str(summary.get("probe_auxiliary", "none")).casefold()
            fusion = str(summary.get("auxiliary_fusion", "residual")).casefold()
            preservation = float(summary.get("auxiliary_preservation_weight", 0.0))
            residual_consistency = float(summary.get("auxiliary_residual_consistency_weight", 0.0))
            gate_supervision = float(summary.get("auxiliary_gate_supervision_weight", 0.0))
            if (
                auxiliary != expected_spec.auxiliary
                or fusion != expected_spec.fusion
                or not np.isclose(preservation, expected_spec.preservation_weight)
                or not np.isclose(residual_consistency, expected_spec.residual_consistency_weight)
                or not np.isclose(gate_supervision, expected_spec.gate_supervision_weight)
            ):
                raise ValueError(f"{path} does not match the locked {method} specification")
            if method in NEW_METHODS:
                if list(summary.get("auxiliary_target_classes", [])) != [2, 3]:
                    raise ValueError(f"{path} changed the locked bilateral target classes")
                if not np.isclose(
                    float(summary.get("auxiliary_gate_initial_probability", float("nan"))),
                    0.25,
                ):
                    raise ValueError(f"{path} changed the locked initial gate probability")
                trainable = int(summary.get("trainable_parameters", 0))
                auxiliary_parameters = int(summary.get("auxiliary_parameters", 0))
                if trainable <= 0 or auxiliary_parameters <= 0:
                    raise ValueError(f"{path} did not record a trainable gated branch")
                new_parameter_counts.add((trainable, auxiliary_parameters))
                for key in (
                    "validation_auxiliary_preservation_loss",
                    "validation_auxiliary_consistency_loss",
                    "validation_auxiliary_gate_target_mean",
                    "validation_auxiliary_gate_nontarget_mean",
                    "validation_auxiliary_gate_supervision_loss",
                ):
                    if not np.isfinite(float(summary.get(key, float("nan")))):
                        raise ValueError(f"{path} is missing finite gate diagnostics: {key}")
            else:
                trainable = int(summary.get("trainable_parameters", 0))
                auxiliary_parameters = int(summary.get("auxiliary_parameters", 0))
            if auxiliary == "gqba_odd_even":
                invariant_error = float(summary.get("auxiliary_reference_max_abs_diff", float("inf")))
                if not np.isfinite(invariant_error) or invariant_error > 1e-4:
                    raise ValueError(f"{path} failed the reference-invariance diagnostic")
            metadata = summary.get("encoder_metadata", {})
            revisions.add(
                (
                    str(metadata.get("model_revision")),
                    str(metadata.get("position_model_revision")),
                )
            )
            fingerprints.add(str(summary.get("dataset_fingerprint")))
            frames[method][probe_seed] = by_view
            rows.extend(
                _metric_row(method, path, probe_seed, view, by_view[view]) for view in EVALUATION_VIEWS
            )
            manifest_rows.append(
                {
                    "method": method,
                    "run_dir": str(path),
                    "probe_seed": probe_seed,
                    "probe_auxiliary": auxiliary,
                    "auxiliary_fusion": fusion,
                    "auxiliary_preservation_weight": preservation,
                    "auxiliary_residual_consistency_weight": residual_consistency,
                    "auxiliary_gate_supervision_weight": gate_supervision,
                    "trainable_parameters": trainable,
                    "auxiliary_parameters": auxiliary_parameters,
                    "validation_gate_target_mean": summary.get("validation_auxiliary_gate_target_mean"),
                    "validation_gate_nontarget_mean": summary.get("validation_auxiliary_gate_nontarget_mean"),
                }
            )
        if observed != expected:
            raise ValueError(f"{method} has seeds {sorted(observed)}, expected {sorted(expected)}")

    if len(new_parameter_counts) != 1:
        raise ValueError(
            "All gated candidate and control arms must have identical parameter counts: "
            f"{sorted(new_parameter_counts)}"
        )
    if len(fingerprints) != 1 or "None" in fingerprints or len(revisions) != 1:
        raise ValueError("GSRA runs do not share one dataset and immutable REVE revision")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_seed = pd.DataFrame(rows).sort_values(["method", "probe_seed", "test_view"])
    by_seed.to_csv(output / "gsra_metrics_by_seed.csv", index=False)
    manifest = pd.DataFrame(manifest_rows).sort_values(["method", "probe_seed"])
    manifest.to_csv(output / "gsra_manifest.csv", index=False)

    aggregate = (
        by_seed.groupby(["method", "test_view"], as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            worst_class_recall_mean=("worst_class_recall", "mean"),
            class_0_recall_mean=("class_0_recall", "mean"),
            class_1_recall_mean=("class_1_recall", "mean"),
            class_2_recall_mean=("class_2_recall", "mean"),
            class_3_recall_mean=("class_3_recall", "mean"),
        )
        .sort_values(["method", "test_view"])
    )
    aggregate.to_csv(output / "gsra_metrics_summary.csv", index=False)

    comparisons: list[dict[str, Any]] = []

    def add_bacc(candidate: str, baseline: str, views: tuple[str, ...], evaluation: str, offset: int):
        result = _hierarchical_bacc_delta_views(
            frames[candidate],
            frames[baseline],
            views=views,
            n_resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed + offset,
        )
        comparisons.append(
            {
                "metric": "balanced_accuracy",
                "evaluation": evaluation,
                "candidate": candidate,
                "baseline": baseline,
                **result,
            }
        )

    add_bacc("gsra", "joint_multiview_ce", PRIMARY_VIEWS, "native16_reference_mean", 1)
    add_bacc("gsra", "gated_spectral_control", PRIMARY_VIEWS, "native16_reference_mean", 2)
    add_bacc("gsra", "gqba_gated_ce", PRIMARY_VIEWS, "native16_reference_mean", 3)
    add_bacc("gsra", "gqba_odd_even", PRIMARY_VIEWS, "native16_reference_mean", 4)
    add_bacc("gsra", "joint_multiview_ce", ("car",), "clean_car", 5)
    add_bacc("gsra", "joint_multiview_ce", NATIVE32_VIEWS, "native32_reference_mean", 6)

    for class_index in range(4):
        result = _hierarchical_recall_delta_views(
            frames["gsra"],
            frames["joint_multiview_ce"],
            views=PRIMARY_VIEWS,
            class_index=class_index,
            n_resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed + 100 + class_index,
        )
        comparisons.append(
            {
                "metric": f"class_{class_index}_recall",
                "evaluation": "native16_reference_mean",
                "candidate": "gsra",
                "baseline": "joint_multiview_ce",
                **result,
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output / "gsra_pairwise_bootstrap.csv", index=False)

    method_rows: list[dict[str, Any]] = []
    for method in GSRA_SPECS:
        selected = aggregate.loc[aggregate["method"] == method].set_index("test_view")
        row: dict[str, Any] = {
            "method": method,
            "clean_car_bacc": float(selected.loc["car", "balanced_accuracy_mean"]),
            "native32_reference_bacc_mean": float(
                selected.loc[list(NATIVE32_VIEWS), "balanced_accuracy_mean"].mean()
            ),
            "native16_reference_bacc_mean": float(
                selected.loc[list(PRIMARY_VIEWS), "balanced_accuracy_mean"].mean()
            ),
        }
        for class_index in range(4):
            row[f"native16_class_{class_index}_recall_mean"] = float(
                selected.loc[list(PRIMARY_VIEWS), f"class_{class_index}_recall_mean"].mean()
            )
        method_rows.append(row)
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(output / "gsra_method_summary.csv", index=False)

    def comparison(candidate: str, baseline: str, metric: str, evaluation: str) -> pd.Series:
        return comparison_frame.loc[
            comparison_frame["candidate"].eq(candidate)
            & comparison_frame["baseline"].eq(baseline)
            & comparison_frame["metric"].eq(metric)
            & comparison_frame["evaluation"].eq(evaluation)
        ].iloc[0]

    primary = comparison("gsra", "joint_multiview_ce", "balanced_accuracy", "native16_reference_mean")
    capacity = comparison("gsra", "gated_spectral_control", "balanced_accuracy", "native16_reference_mean")
    objective = comparison("gsra", "gqba_gated_ce", "balanced_accuracy", "native16_reference_mean")
    clean = comparison("gsra", "joint_multiview_ce", "balanced_accuracy", "clean_car")
    native32 = comparison("gsra", "joint_multiview_ce", "balanced_accuracy", "native32_reference_mean")
    class_comparisons = [
        comparison(
            "gsra",
            "joint_multiview_ce",
            f"class_{class_index}_recall",
            "native16_reference_mean",
        )
        for class_index in range(4)
    ]
    gsra_manifest = manifest.loc[manifest["method"] == "gsra"]
    gate_selectivity = float(
        (
            gsra_manifest["validation_gate_target_mean"] - gsra_manifest["validation_gate_nontarget_mean"]
        ).mean()
    )
    gates = {
        "primary_ci_above_zero_vs_joint": bool(primary["ci_lower"] > 0.0),
        "beats_gated_capacity_control": bool(capacity["ci_lower"] > 0.0),
        "beats_gqba_gated_ce_ablation": bool(objective["ci_lower"] > 0.0),
        "clean_car_noninferior_to_joint": bool(
            clean["candidate_minus_baseline"] >= -0.01 and clean["ci_lower"] >= -0.01
        ),
        "native32_noninferior_to_joint": bool(
            native32["candidate_minus_baseline"] >= -0.01 and native32["ci_lower"] >= -0.01
        ),
        "all_native16_classes_preserved": bool(
            all(item["candidate_minus_baseline"] >= -0.01 for item in class_comparisons)
        ),
        "both_fists_gain_at_least_0p03_with_positive_ci": bool(
            class_comparisons[2]["candidate_minus_baseline"] >= 0.03
            and class_comparisons[2]["ci_lower"] > 0.0
        ),
        "validation_gate_prefers_target_class": bool(gate_selectivity > 0.0),
    }
    decision = {
        "stage": "GaugeEEG GSRA locked development screen",
        "status": "development_method_screen_only",
        "primary_metric": "mean BAcc over native16@{CAR,Cz,Pz,Fz}",
        "proposed_method": "gsra",
        "strong_baseline": "joint_multiview_ce",
        "capacity_control": "gated_spectral_control",
        "architecture_ablation": "gqba_gated_ce",
        "prior_ungated_ablation": "gqba_odd_even",
        "physionetmi_is_globally_untouched_test": False,
        "external_dataset_required_for_confirmation": True,
        "parameter_count_match": sorted(new_parameter_counts)[0],
        "validation_gate_target_minus_nontarget": gate_selectivity,
        "gates": gates,
        "hypothesis_supported": bool(all(gates.values())),
        "interpretation_if_failed": ("Retain joint_multiview_ce; do not tune GSRA on subjects 71--89."),
    }
    with (output / "gsra_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    return method_summary

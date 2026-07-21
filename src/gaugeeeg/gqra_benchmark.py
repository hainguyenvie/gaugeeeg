"""Locked development audit for Gauge-Quotient Representation Alignment."""

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
from .gsra_benchmark import _hierarchical_recall_delta_views

PRIMARY_VIEWS = tuple(view for view in EVALUATION_VIEWS if view.startswith("native16@"))
NATIVE32_VIEWS = tuple(view for view in EVALUATION_VIEWS if view.startswith("native32@"))


@dataclass(frozen=True)
class GQRASpec:
    auxiliary: str
    fusion: str
    preservation_weight: float
    residual_consistency_weight: float
    gate_supervision_weight: float
    contrastive_weight: float
    bilaterality_weight: float
    temperature: float


GQRA_SPECS: dict[str, GQRASpec] = {
    "joint_multiview_ce": GQRASpec("none", "residual", 0.0, 0.0, 0.0, 0.0, 0.0, 0.1),
    "gsra": GQRASpec("gqba_odd_even", "gated_residual", 1.0, 0.1, 0.1, 0.0, 0.0, 0.1),
    "film_spectral_control": GQRASpec("spectral_capacity_control", "film", 0.0, 0.0, 0.0, 0.0, 0.0, 0.1),
    "gqba_film_ce": GQRASpec("gqba_odd_even", "film", 0.0, 0.0, 0.0, 0.0, 0.0, 0.1),
    "gqra": GQRASpec("gqba_odd_even", "film", 0.0, 0.0, 0.0, 0.1, 0.2, 0.1),
}
NEW_METHODS = ("film_spectral_control", "gqba_film_ce", "gqra")


def _parse_specs(run_specs: list[str]) -> dict[str, list[Path]]:
    grouped = {method: [] for method in GQRA_SPECS}
    for item in run_specs:
        if "=" not in item:
            raise ValueError(f"Invalid run specification {item!r}; expected method=path")
        method, path = item.split("=", maxsplit=1)
        if method not in grouped:
            raise ValueError(f"Unknown GQRA method {method!r}; expected {sorted(grouped)}")
        grouped[method].append(Path(path))
    missing = [method for method, paths in grouped.items() if not paths]
    if missing:
        raise ValueError(f"Missing required GQRA runs: {missing}")
    return grouped


def analyze_gqra_benchmark(
    run_specs: list[str],
    output_dir: str | Path,
    *,
    expected_seeds: list[int] | tuple[int, ...] = (7, 21, 42),
    bootstrap_resamples: int = 10_000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260724,
) -> pd.DataFrame:
    """Validate the matched representation screen and freeze its decision."""

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
        spec = GQRA_SPECS[method]
        for path in paths:
            summary, by_view = _validate_run(method, path, spec=common_spec)
            probe_seed = int(summary["probe_seed"])
            if probe_seed in observed:
                raise ValueError(f"Duplicate seed {probe_seed} for {method}")
            observed.add(probe_seed)
            actual = GQRASpec(
                str(summary.get("probe_auxiliary", "none")).casefold(),
                str(summary.get("auxiliary_fusion", "residual")).casefold(),
                float(summary.get("auxiliary_preservation_weight", 0.0)),
                float(summary.get("auxiliary_residual_consistency_weight", 0.0)),
                float(summary.get("auxiliary_gate_supervision_weight", 0.0)),
                float(summary.get("representation_contrastive_weight", 0.0)),
                float(summary.get("representation_bilaterality_weight", 0.0)),
                float(summary.get("representation_temperature", 0.1)),
            )
            if actual != spec:
                raise ValueError(f"{path} does not match the locked {method} specification")
            if method in NEW_METHODS:
                if list(summary.get("auxiliary_target_classes", [])) != [2, 3]:
                    raise ValueError(f"{path} changed the locked bilaterality target")
                trainable = int(summary.get("trainable_parameters", 0))
                auxiliary_parameters = int(summary.get("auxiliary_parameters", 0))
                if trainable <= 0 or auxiliary_parameters <= 0:
                    raise ValueError(f"{path} did not record a trainable FiLM branch")
                new_parameter_counts.add((trainable, auxiliary_parameters))
                for key in (
                    "validation_representation_alignment_loss",
                    "validation_representation_class_margin",
                    "validation_representation_bilaterality_balanced_accuracy",
                ):
                    if not np.isfinite(float(summary.get(key, float("nan")))):
                        raise ValueError(f"{path} is missing finite representation diagnostic: {key}")
            else:
                trainable = int(summary.get("trainable_parameters", 0))
                auxiliary_parameters = int(summary.get("auxiliary_parameters", 0))
            if spec.auxiliary == "gqba_odd_even":
                invariant_error = float(summary.get("auxiliary_reference_max_abs_diff", float("inf")))
                if not np.isfinite(invariant_error) or invariant_error > 1e-4:
                    raise ValueError(f"{path} failed the reference-invariance diagnostic")
            metadata = summary.get("encoder_metadata", {})
            revisions.add((str(metadata.get("model_revision")), str(metadata.get("position_model_revision"))))
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
                    "probe_auxiliary": spec.auxiliary,
                    "auxiliary_fusion": spec.fusion,
                    "representation_contrastive_weight": spec.contrastive_weight,
                    "representation_bilaterality_weight": spec.bilaterality_weight,
                    "representation_temperature": spec.temperature,
                    "trainable_parameters": trainable,
                    "auxiliary_parameters": auxiliary_parameters,
                    "validation_representation_alignment_loss": summary.get(
                        "validation_representation_alignment_loss"
                    ),
                    "validation_representation_class_margin": summary.get(
                        "validation_representation_class_margin"
                    ),
                    "validation_representation_bilaterality_balanced_accuracy": summary.get(
                        "validation_representation_bilaterality_balanced_accuracy"
                    ),
                }
            )
        if observed != expected:
            raise ValueError(f"{method} has seeds {sorted(observed)}, expected {sorted(expected)}")

    if len(new_parameter_counts) != 1:
        raise ValueError(
            "All FiLM candidate and control arms must have identical parameter counts: "
            f"{sorted(new_parameter_counts)}"
        )
    if len(fingerprints) != 1 or "None" in fingerprints or len(revisions) != 1:
        raise ValueError("GQRA runs do not share one dataset and immutable REVE revision")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_seed = pd.DataFrame(rows).sort_values(["method", "probe_seed", "test_view"])
    by_seed.to_csv(output / "gqra_metrics_by_seed.csv", index=False)
    manifest = pd.DataFrame(manifest_rows).sort_values(["method", "probe_seed"])
    manifest.to_csv(output / "gqra_manifest.csv", index=False)
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
    aggregate.to_csv(output / "gqra_metrics_summary.csv", index=False)

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

    add_bacc("gqra", "joint_multiview_ce", PRIMARY_VIEWS, "native16_reference_mean", 1)
    add_bacc("gqra", "film_spectral_control", PRIMARY_VIEWS, "native16_reference_mean", 2)
    add_bacc("gqra", "gqba_film_ce", PRIMARY_VIEWS, "native16_reference_mean", 3)
    add_bacc("gqra", "gsra", PRIMARY_VIEWS, "native16_reference_mean", 4)
    add_bacc("gqra", "joint_multiview_ce", ("car",), "clean_car", 5)
    add_bacc("gqra", "joint_multiview_ce", NATIVE32_VIEWS, "native32_reference_mean", 6)
    for class_index in range(4):
        result = _hierarchical_recall_delta_views(
            frames["gqra"],
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
                "candidate": "gqra",
                "baseline": "joint_multiview_ce",
                **result,
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output / "gqra_pairwise_bootstrap.csv", index=False)

    method_rows: list[dict[str, Any]] = []
    for method in GQRA_SPECS:
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
    method_summary.to_csv(output / "gqra_method_summary.csv", index=False)

    def comparison(candidate: str, baseline: str, metric: str, evaluation: str) -> pd.Series:
        return comparison_frame.loc[
            comparison_frame["candidate"].eq(candidate)
            & comparison_frame["baseline"].eq(baseline)
            & comparison_frame["metric"].eq(metric)
            & comparison_frame["evaluation"].eq(evaluation)
        ].iloc[0]

    primary = comparison("gqra", "joint_multiview_ce", "balanced_accuracy", "native16_reference_mean")
    capacity = comparison("gqra", "film_spectral_control", "balanced_accuracy", "native16_reference_mean")
    feature = comparison("gqra", "gqba_film_ce", "balanced_accuracy", "native16_reference_mean")
    prior = comparison("gqra", "gsra", "balanced_accuracy", "native16_reference_mean")
    clean = comparison("gqra", "joint_multiview_ce", "balanced_accuracy", "clean_car")
    native32 = comparison("gqra", "joint_multiview_ce", "balanced_accuracy", "native32_reference_mean")
    class_comparisons = [
        comparison("gqra", "joint_multiview_ce", f"class_{index}_recall", "native16_reference_mean")
        for index in range(4)
    ]
    candidate_manifest = manifest.loc[manifest["method"] == "gqra"]
    ablation_manifest = manifest.loc[manifest["method"] == "gqba_film_ce"]
    candidate_alignment = float(candidate_manifest["validation_representation_alignment_loss"].mean())
    ablation_alignment = float(ablation_manifest["validation_representation_alignment_loss"].mean())
    candidate_bilaterality = float(
        candidate_manifest["validation_representation_bilaterality_balanced_accuracy"].mean()
    )
    gates = {
        "primary_ci_above_zero_vs_joint": bool(primary["ci_lower"] > 0.0),
        "beats_matched_spectral_control": bool(capacity["ci_lower"] > 0.0),
        "beats_gqba_film_ce_ablation": bool(feature["ci_lower"] > 0.0),
        "beats_prior_gsra": bool(prior["ci_lower"] > 0.0),
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
        "representation_alignment_improves_over_ce": bool(candidate_alignment < ablation_alignment),
        "validation_bilaterality_bacc_at_least_0p55": bool(candidate_bilaterality >= 0.55),
    }
    decision = {
        "stage": "GaugeEEG GQRA locked development screen",
        "status": "development_method_screen_only",
        "primary_metric": "mean BAcc over native16@{CAR,Cz,Pz,Fz}",
        "proposed_method": "gqra",
        "strong_baseline": "joint_multiview_ce",
        "capacity_control": "film_spectral_control",
        "feature_and_objective_ablation": "gqba_film_ce",
        "prior_failed_method": "gsra",
        "physionetmi_is_globally_untouched_test": False,
        "external_dataset_required_for_confirmation": True,
        "parameter_count_match": list(next(iter(new_parameter_counts))),
        "validation_representation_alignment": candidate_alignment,
        "validation_ce_ablation_alignment": ablation_alignment,
        "validation_bilaterality_balanced_accuracy": candidate_bilaterality,
        "gates": gates,
        "hypothesis_supported": bool(all(gates.values())),
        "interpretation_if_failed": ("Retain joint_multiview_ce; do not tune GQRA on subjects 71--89."),
    }
    with (output / "gqra_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    return aggregate

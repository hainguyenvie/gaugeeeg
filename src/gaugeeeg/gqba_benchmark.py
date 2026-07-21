"""Locked development audit for the Gauge-Quotient Bilateral Adapter."""

from __future__ import annotations

import json
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

GQBA_MODES: dict[str, str] = {
    "car_only": "none",
    "joint_multiview_ce": "none",
    "spectral_capacity_control": "spectral_capacity_control",
    "gqba_odd": "gqba_odd",
    "gqba_odd_even": "gqba_odd_even",
}
PRIMARY_VIEWS = tuple(view for view in EVALUATION_VIEWS if view.startswith("native16@"))


def _parse_specs(run_specs: list[str]) -> dict[str, list[Path]]:
    grouped = {method: [] for method in GQBA_MODES}
    for item in run_specs:
        if "=" not in item:
            raise ValueError(f"Invalid run specification {item!r}; expected method=path")
        method, path = item.split("=", maxsplit=1)
        if method not in grouped:
            raise ValueError(f"Unknown GQBA method {method!r}; expected {sorted(grouped)}")
        grouped[method].append(Path(path))
    missing = [method for method, paths in grouped.items() if not paths]
    if missing:
        raise ValueError(f"Missing required GQBA runs: {missing}")
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
    class_index: int,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    probe_seeds = sorted(candidate)
    if probe_seeds != sorted(baseline):
        raise ValueError("Candidate and baseline probe seeds differ")
    point = np.mean(
        [
            np.mean(
                [
                    _class_recall(candidate[probe_seed][view], class_index)
                    - _class_recall(baseline[probe_seed][view], class_index)
                    for view in PRIMARY_VIEWS
                ]
            )
            for probe_seed in probe_seeds
        ]
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for draw_index in range(n_resamples):
        seed_values = []
        for sampled_seed in rng.choice(probe_seeds, size=len(probe_seeds), replace=True):
            probe_seed = int(sampled_seed)
            first = candidate[probe_seed][PRIMARY_VIEWS[0]]
            subjects = first["subject_id"].unique()
            sampled_subjects = rng.choice(subjects, size=subjects.size, replace=True)
            positions = np.concatenate(
                [np.flatnonzero(first["subject_id"].to_numpy() == subject) for subject in sampled_subjects]
            )
            view_values = []
            for view in PRIMARY_VIEWS:
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
        "candidate_minus_baseline": float(point),
        "bootstrap_mean": float(draws.mean()),
        "ci_lower": float(np.quantile(draws, alpha)),
        "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
    }


def analyze_gqba_benchmark(
    run_specs: list[str],
    output_dir: str | Path,
    *,
    expected_seeds: list[int] | tuple[int, ...] = (7, 21, 42),
    bootstrap_resamples: int = 10_000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260722,
) -> pd.DataFrame:
    """Validate the matched GQBA screen and write its frozen decision."""

    grouped = _parse_specs(run_specs)
    expected = {int(seed) for seed in expected_seeds}
    frames: dict[str, dict[int, dict[str, pd.DataFrame]]] = {}
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    candidate_parameter_counts: set[tuple[int, int]] = set()
    fingerprints: set[str] = set()
    revisions: set[tuple[str, str]] = set()

    for method, paths in grouped.items():
        frames[method] = {}
        observed: set[int] = set()
        training_views = ("car",) if method == "car_only" else JOINT_TRAINING_VIEWS
        objective = "car_only" if method == "car_only" else "multi_view_ce"
        spec = BaselineSpec(training_views, objective)
        for path in paths:
            summary, by_view = _validate_run(method, path, spec=spec)
            probe_seed = int(summary["probe_seed"])
            if probe_seed in observed:
                raise ValueError(f"Duplicate seed {probe_seed} for {method}")
            observed.add(probe_seed)
            mode = str(summary.get("probe_auxiliary", "none")).casefold()
            if mode != GQBA_MODES[method]:
                raise ValueError(f"{path} uses auxiliary mode {mode!r}, expected {GQBA_MODES[method]!r}")
            trainable = int(summary.get("trainable_parameters", 0))
            auxiliary = int(summary.get("auxiliary_parameters", 0))
            if method not in {"car_only", "joint_multiview_ce"}:
                if trainable <= 0 or auxiliary <= 0:
                    raise ValueError(f"{path} did not record a trainable auxiliary branch")
                candidate_parameter_counts.add((trainable, auxiliary))
            if method in {"gqba_odd", "gqba_odd_even"}:
                invariant_error = float(summary.get("auxiliary_reference_max_abs_diff", float("inf")))
                if not np.isfinite(invariant_error) or invariant_error > 1e-4:
                    raise ValueError(
                        f"{path} failed the exact reference-invariance diagnostic: {invariant_error}"
                    )
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
                    "probe_auxiliary": mode,
                    "trainable_parameters": trainable,
                    "auxiliary_parameters": auxiliary,
                }
            )
        if observed != expected:
            raise ValueError(f"{method} has seeds {sorted(observed)}, expected {sorted(expected)}")

    if len(candidate_parameter_counts) != 1:
        raise ValueError(
            "Capacity control, odd, and odd+even must have identical parameter counts: "
            f"{sorted(candidate_parameter_counts)}"
        )
    if len(fingerprints) != 1 or "None" in fingerprints or len(revisions) != 1:
        raise ValueError("GQBA runs do not share one dataset and immutable REVE revision")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_seed = pd.DataFrame(rows).sort_values(["method", "probe_seed", "test_view"])
    by_seed.to_csv(output / "gqba_metrics_by_seed.csv", index=False)
    pd.DataFrame(manifest_rows).sort_values(["method", "probe_seed"]).to_csv(
        output / "gqba_manifest.csv", index=False
    )

    aggregate = (
        by_seed.groupby(["method", "test_view"], as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            worst_class_recall_mean=("worst_class_recall", "mean"),
            both_fists_recall_mean=("class_2_recall", "mean"),
        )
        .sort_values(["method", "test_view"])
    )
    aggregate.to_csv(output / "gqba_metrics_summary.csv", index=False)

    comparisons: list[dict[str, Any]] = []
    candidates = ("spectral_capacity_control", "gqba_odd", "gqba_odd_even")
    for candidate_index, candidate in enumerate(candidates):
        baselines = ("car_only", "joint_multiview_ce", "spectral_capacity_control")
        for baseline_index, baseline in enumerate(baselines):
            if candidate == baseline:
                continue
            views = ("car",) if baseline == "car_only" else PRIMARY_VIEWS
            result = _hierarchical_bacc_delta_views(
                frames[candidate],
                frames[baseline],
                views=views,
                n_resamples=bootstrap_resamples,
                confidence=bootstrap_confidence,
                seed=bootstrap_seed + 10 * candidate_index + baseline_index,
            )
            comparisons.append(
                {
                    "metric": "balanced_accuracy",
                    "evaluation": "clean_car" if views == ("car",) else "native16_reference_mean",
                    "candidate": candidate,
                    "baseline": baseline,
                    **result,
                }
            )
        recall_result = _hierarchical_recall_delta_views(
            frames[candidate],
            frames["joint_multiview_ce"],
            class_index=2,
            n_resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed + 100 + candidate_index,
        )
        comparisons.append(
            {
                "metric": "both_fists_recall",
                "evaluation": "native16_reference_mean",
                "candidate": candidate,
                "baseline": "joint_multiview_ce",
                **recall_result,
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output / "gqba_pairwise_bootstrap.csv", index=False)

    method_rows: list[dict[str, Any]] = []
    for method in GQBA_MODES:
        selected = aggregate.loc[aggregate["method"] == method].set_index("test_view")
        method_rows.append(
            {
                "method": method,
                "clean_car_bacc": float(selected.loc["car", "balanced_accuracy_mean"]),
                "native16_reference_bacc_mean": float(
                    selected.loc[list(PRIMARY_VIEWS), "balanced_accuracy_mean"].mean()
                ),
                "native16_both_fists_recall_mean": float(
                    selected.loc[list(PRIMARY_VIEWS), "both_fists_recall_mean"].mean()
                ),
                "native16_worst_class_recall": float(
                    selected.loc[list(PRIMARY_VIEWS), "worst_class_recall_mean"].min()
                ),
            }
        )
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(output / "gqba_method_summary.csv", index=False)

    def comparison(candidate: str, baseline: str, metric: str, evaluation: str) -> pd.Series:
        return comparison_frame.loc[
            comparison_frame["candidate"].eq(candidate)
            & comparison_frame["baseline"].eq(baseline)
            & comparison_frame["metric"].eq(metric)
            & comparison_frame["evaluation"].eq(evaluation)
        ].iloc[0]

    proposed = method_summary.set_index("method").loc["gqba_odd_even"]
    joint = method_summary.set_index("method").loc["joint_multiview_ce"]
    clean = comparison("gqba_odd_even", "car_only", "balanced_accuracy", "clean_car")
    primary = comparison(
        "gqba_odd_even",
        "joint_multiview_ce",
        "balanced_accuracy",
        "native16_reference_mean",
    )
    capacity = comparison(
        "gqba_odd_even",
        "spectral_capacity_control",
        "balanced_accuracy",
        "native16_reference_mean",
    )
    both_fists = comparison(
        "gqba_odd_even",
        "joint_multiview_ce",
        "both_fists_recall",
        "native16_reference_mean",
    )
    gates = {
        "primary_ci_above_zero_vs_joint": bool(primary["ci_lower"] > 0.0),
        "beats_matched_capacity_control": bool(capacity["ci_lower"] > 0.0),
        "clean_car_noninferior_to_car_only": bool(
            clean["candidate_minus_baseline"] >= -0.01 and clean["ci_lower"] >= -0.01
        ),
        "both_fists_recall_gain_at_least_0p03": bool(both_fists["candidate_minus_baseline"] >= 0.03),
        "worst_class_recall_preserved": bool(
            proposed["native16_worst_class_recall"] >= joint["native16_worst_class_recall"] - 0.01
        ),
    }
    decision = {
        "stage": "GaugeEEG GQBA matched-capacity development screen",
        "status": "development_method_screen_only",
        "primary_metric": "mean BAcc over native16@{CAR,Cz,Pz,Fz}",
        "proposed_method": "gqba_odd_even",
        "strong_baseline": "joint_multiview_ce",
        "capacity_control": "spectral_capacity_control",
        "physionetmi_is_globally_untouched_test": False,
        "external_dataset_required_for_confirmation": True,
        "parameter_count_match": sorted(candidate_parameter_counts)[0],
        "gates": gates,
        "hypothesis_supported": bool(all(gates.values())),
        "interpretation_if_failed": (
            "Retain joint_multiview_ce; do not claim GQBA or tune its rule on the audit subjects."
        ),
    }
    with (output / "gqba_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    return method_summary

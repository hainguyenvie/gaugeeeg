"""Post-hoc E13 audit against the strongest E12 baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .class_safeguard import (
    BASELINE_METHOD,
    E11_METHOD,
    SAFE_METHOD,
    TOPOLOGY_METHOD,
)
from .prior_stress import _json_value


RMSE_METRIC = "target_bias_rmse"
BACC_METRIC = "balanced_accuracy"
GAP_METRIC = "max_class_recall_gap_to_car"
METRICS = (RMSE_METRIC, BACC_METRIC, GAP_METRIC)
CONTROL_BASELINES = (BASELINE_METHOD, E11_METHOD, TOPOLOGY_METHOD)
STRONG_BASELINES = (E11_METHOD, TOPOLOGY_METHOD)


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    batch_size: int
    condition: str | None = None
    condition_prefix: str | None = None
    dominant_class: int | None = None
    class_name: str | None = None


def _select_metrics(frame: pd.DataFrame, spec: ComparisonSpec) -> pd.DataFrame:
    selected = frame.loc[frame["batch_size"] == spec.batch_size]
    if spec.condition is not None:
        selected = selected.loc[selected["condition"] == spec.condition]
    if spec.condition_prefix is not None:
        selected = selected.loc[
            selected["condition"].str.startswith(spec.condition_prefix)
        ]
    if selected.empty:
        raise ValueError(f"No metric rows found for {spec.name}")
    return selected


def _metric_direction(metric: str) -> str:
    return "higher" if metric == BACC_METRIC else "lower"


def _paired_cluster_matrices(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str], bool]:
    selected = frame.loc[frame["method"].isin([candidate, baseline])]
    grouped = (
        selected.groupby(
            ["repeat", "held_out_reference", "method"], as_index=False
        )[metric]
        .mean()
    )
    candidate_rows = grouped.loc[grouped["method"] == candidate]
    baseline_rows = grouped.loc[grouped["method"] == baseline]
    if candidate_rows.empty or baseline_rows.empty:
        raise RuntimeError(
            f"Incomplete comparison for {candidate} versus {baseline}"
        )

    repeats = sorted(candidate_rows["repeat"].astype(int).unique().tolist())
    references = sorted(candidate_rows["held_out_reference"].astype(str).unique())
    candidate_pivot = candidate_rows.pivot(
        index="repeat", columns="held_out_reference", values=metric
    ).reindex(index=repeats, columns=references)
    if candidate_pivot.isna().any().any():
        raise RuntimeError("Candidate repeat/reference grid is incomplete")

    baseline_repeats = sorted(
        baseline_rows["repeat"].astype(int).unique().tolist()
    )
    static_baseline = len(baseline_repeats) == 1 and len(repeats) > 1
    if static_baseline:
        baseline_vector = (
            baseline_rows.groupby("held_out_reference")[metric]
            .mean()
            .reindex(references)
        )
        if baseline_vector.isna().any():
            raise RuntimeError("Static baseline reference grid is incomplete")
        baseline_values = np.broadcast_to(
            baseline_vector.to_numpy(dtype=np.float64),
            candidate_pivot.shape,
        ).copy()
    else:
        if baseline_repeats != repeats:
            raise RuntimeError("Dynamic baseline repeat grid does not match candidate")
        baseline_pivot = baseline_rows.pivot(
            index="repeat", columns="held_out_reference", values=metric
        ).reindex(index=repeats, columns=references)
        if baseline_pivot.isna().any().any():
            raise RuntimeError("Dynamic baseline repeat/reference grid is incomplete")
        baseline_values = baseline_pivot.to_numpy(dtype=np.float64)

    return (
        candidate_pivot.to_numpy(dtype=np.float64),
        baseline_values,
        repeats,
        references,
        static_baseline,
    )


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    metric: str,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a paired method delta over repeats and reference identities."""

    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    if n_resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap settings")
    candidate_values, baseline_values, repeats, references, static = (
        _paired_cluster_matrices(
            frame,
            candidate=candidate,
            baseline=baseline,
            metric=metric,
        )
    )
    delta = candidate_values - baseline_values
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        repeat_index = rng.integers(0, delta.shape[0], size=delta.shape[0])
        reference_index = rng.integers(
            0, delta.shape[1], size=delta.shape[1]
        )
        samples[index] = delta[
            np.ix_(repeat_index, reference_index)
        ].mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "candidate": candidate,
        "baseline": baseline,
        "candidate_cluster_mean": float(candidate_values.mean()),
        "baseline_cluster_mean": float(baseline_values.mean()),
        "candidate_minus_baseline": float(delta.mean()),
        "ci_lower": float(np.quantile(samples, alpha)),
        "ci_upper": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "n_repeats": len(repeats),
        "n_reference_clusters": len(references),
        "static_baseline_broadcast_across_repeats": static,
    }


def _decorate_comparison(
    row: dict[str, Any],
    *,
    max_bacc_loss: float,
    max_gap_increase: float,
) -> dict[str, Any]:
    metric = str(row["metric"])
    delta = float(row["candidate_minus_baseline"])
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    direction = _metric_direction(metric)
    if metric == RMSE_METRIC:
        point_improves = delta < 0.0
        interval_confirms_improvement = upper < 0.0
        point_within_tolerance = delta <= 0.0
        interval_confirms_noninferiority = upper <= 0.0
        material_harm_detected = lower > 0.0
        tolerance = 0.0
    elif metric == BACC_METRIC:
        point_improves = delta > 0.0
        interval_confirms_improvement = lower > 0.0
        point_within_tolerance = delta >= -max_bacc_loss
        interval_confirms_noninferiority = lower >= -max_bacc_loss
        material_harm_detected = upper < -max_bacc_loss
        tolerance = max_bacc_loss
    else:
        point_improves = delta < 0.0
        interval_confirms_improvement = upper < 0.0
        point_within_tolerance = delta <= max_gap_increase
        interval_confirms_noninferiority = upper <= max_gap_increase
        material_harm_detected = lower > max_gap_increase
        tolerance = max_gap_increase
    return {
        **row,
        "direction": direction,
        "noninferiority_tolerance": tolerance,
        "point_improves": bool(point_improves),
        "interval_confirms_improvement": bool(
            interval_confirms_improvement
        ),
        "point_within_tolerance": bool(point_within_tolerance),
        "interval_confirms_noninferiority": bool(
            interval_confirms_noninferiority
        ),
        "material_harm_detected": bool(material_harm_detected),
    }


def _comparison_specs(summary: dict[str, Any]) -> list[ComparisonSpec]:
    primary_size = int(summary["primary_batch_size"])
    stress_size = int(summary["stress_batch_size"])
    class_names = list(summary["class_names"])
    return [
        ComparisonSpec(
            "primary_random",
            primary_size,
            condition="random",
        ),
        ComparisonSpec(
            "balanced_stress_size",
            stress_size,
            condition="balanced",
        ),
        ComparisonSpec(
            "severe_skew",
            stress_size,
            condition_prefix="skew_0.7_",
        ),
        *[
            ComparisonSpec(
                f"severe_{class_name}",
                stress_size,
                condition=f"skew_0.7_class_{class_index}",
                dominant_class=class_index,
                class_name=class_name,
            )
            for class_index, class_name in enumerate(class_names)
        ],
    ]


def _class_audit(pairwise: pd.DataFrame, specs: list[ComparisonSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.dominant_class is None:
            continue
        selected = pairwise.loc[pairwise["comparison"] == spec.name]
        strong = selected.loc[selected["baseline"].isin(STRONG_BASELINES)]
        rmse = strong.loc[strong["metric"] == RMSE_METRIC]
        ranking = rmse.sort_values("baseline_cluster_mean").iloc[0]
        rows.append(
            {
                "comparison": spec.name,
                "dominant_class": spec.dominant_class,
                "class_name": spec.class_name,
                "safe_cluster_mean_rmse": float(
                    rmse["candidate_cluster_mean"].iloc[0]
                ),
                "strongest_baseline": ranking["baseline"],
                "strongest_baseline_cluster_mean_rmse": float(
                    ranking["baseline_cluster_mean"]
                ),
                "safe_minus_strongest_baseline_rmse": float(
                    ranking["candidate_minus_baseline"]
                ),
                "safe_vs_strongest_ci_lower": float(ranking["ci_lower"]),
                "safe_vs_strongest_ci_upper": float(ranking["ci_upper"]),
                "rmse_point_improves_vs_all_strong_baselines": bool(
                    rmse["point_improves"].all()
                ),
                "rmse_intervals_confirm_vs_all_strong_baselines": bool(
                    rmse["interval_confirms_improvement"].all()
                ),
                "any_material_harm_vs_strong_baselines": bool(
                    strong["material_harm_detected"].any()
                ),
                "failing_rmse_baselines": json.dumps(
                    rmse.loc[~rmse["point_improves"], "baseline"].tolist()
                ),
                "material_harm_comparisons": json.dumps(
                    strong.loc[
                        strong["material_harm_detected"],
                        ["metric", "baseline"],
                    ].to_dict(orient="records")
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_strong_baselines(
    e12_output: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_resamples: int = 5000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260719,
    max_bacc_loss: float = 0.01,
    max_gap_increase: float = 0.01,
) -> pd.DataFrame:
    """Audit E12 against strict E11 and topology without refitting anything."""

    if bootstrap_resamples < 1 or not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("Invalid bootstrap settings")
    if max_bacc_loss < 0.0 or max_gap_increase < 0.0:
        raise ValueError("Noninferiority tolerances must be non-negative")

    source = Path(e12_output)
    summary_path = source / "class_safeguard_summary.json"
    metrics_path = source / "class_safeguard_metrics.csv"
    if not summary_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("E12 summary and metrics are required")
    e12_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if e12_summary.get("stage") != "E12 source-only class/operator trust safeguard":
        raise ValueError("Input is not an E12 class-safeguard output")
    split_checks = [
        "source_adaptation_subjects_disjoint",
        "source_evaluation_subjects_disjoint",
        "adaptation_evaluation_subjects_disjoint",
    ]
    failed_splits = [key for key in split_checks if not e12_summary.get(key)]
    if failed_splits:
        raise ValueError(f"E12 split checks failed: {failed_splits}")

    metrics = pd.read_csv(metrics_path)
    required_columns = {
        "condition",
        "batch_size",
        "repeat",
        "held_out_reference",
        "method",
        *METRICS,
    }
    if not required_columns <= set(metrics.columns):
        raise ValueError(
            f"E12 metrics miss columns: {sorted(required_columns - set(metrics))}"
        )
    required_methods = {SAFE_METHOD, *CONTROL_BASELINES}
    if not required_methods <= set(metrics["method"]):
        raise ValueError(
            f"E12 metrics miss methods: {sorted(required_methods - set(metrics['method']))}"
        )

    specs = _comparison_specs(e12_summary)
    comparison_rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        selected = _select_metrics(metrics, spec)
        for baseline_index, baseline in enumerate(CONTROL_BASELINES):
            for metric_index, metric in enumerate(METRICS):
                result = paired_cluster_bootstrap(
                    selected,
                    candidate=SAFE_METHOD,
                    baseline=baseline,
                    metric=metric,
                    n_resamples=bootstrap_resamples,
                    confidence=bootstrap_confidence,
                    seed=(
                        bootstrap_seed
                        + 100 * spec_index
                        + 10 * baseline_index
                        + metric_index
                    ),
                )
                comparison_rows.append(
                    _decorate_comparison(
                        {
                            "comparison": spec.name,
                            "dominant_class": spec.dominant_class,
                            "class_name": spec.class_name,
                            **result,
                        },
                        max_bacc_loss=max_bacc_loss,
                        max_gap_increase=max_gap_increase,
                    )
                )
    pairwise = pd.DataFrame(comparison_rows)
    class_audit = _class_audit(pairwise, specs)

    mean_names = {"primary_random", "balanced_stress_size", "severe_skew"}
    mean_strong = pairwise.loc[
        pairwise["comparison"].isin(mean_names)
        & pairwise["baseline"].isin(STRONG_BASELINES)
    ]
    mean_rmse = mean_strong.loc[mean_strong["metric"] == RMSE_METRIC]
    mean_tasks = mean_strong.loc[mean_strong["metric"].isin([BACC_METRIC, GAP_METRIC])]
    mean_point_supported = bool(mean_rmse["point_improves"].all())
    mean_interval_supported = bool(
        mean_rmse["interval_confirms_improvement"].all()
    )
    mean_task_noninferiority = bool(
        mean_tasks["interval_confirms_noninferiority"].all()
    )
    mean_harm_detected = bool(mean_strong["material_harm_detected"].any())
    class_point_supported = bool(
        class_audit["rmse_point_improves_vs_all_strong_baselines"].all()
    )
    class_interval_supported = bool(
        class_audit[
            "rmse_intervals_confirm_vs_all_strong_baselines"
        ].all()
    )
    class_harm_detected = bool(
        class_audit["any_material_harm_vs_strong_baselines"].any()
    )
    original_gate = bool(
        e12_summary.get("safeguard_supported_for_repeated_seed_confirmation")
    )
    mean_ready = bool(
        original_gate
        and mean_point_supported
        and mean_task_noninferiority
        and not mean_harm_detected
    )
    class_ready = bool(
        mean_ready
        and class_point_supported
        and not class_harm_detected
    )

    summary = {
        "stage": "E13 post-hoc strongest-baseline audit",
        "source_e12_output": str(source),
        "audit_status": "post_hoc_falsification_only",
        "stronger_gate_defined_after_e12_seed7_review": True,
        "result_can_falsify_but_not_confirm_a_paper_claim": True,
        "refitting_performed": False,
        "target_labels_used_for_new_fitting": False,
        "source_adaptation_subjects_disjoint": e12_summary[
            "source_adaptation_subjects_disjoint"
        ],
        "source_evaluation_subjects_disjoint": e12_summary[
            "source_evaluation_subjects_disjoint"
        ],
        "adaptation_evaluation_subjects_disjoint": e12_summary[
            "adaptation_evaluation_subjects_disjoint"
        ],
        "physionet_test_subjects_used": e12_summary[
            "physionet_test_subjects_used"
        ],
        "candidate": SAFE_METHOD,
        "control_baselines": list(CONTROL_BASELINES),
        "predeclared_strong_baselines": list(STRONG_BASELINES),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_confidence": bootstrap_confidence,
        "bootstrap_seed": bootstrap_seed,
        "max_bacc_loss": max_bacc_loss,
        "max_gap_increase": max_gap_increase,
        "n_reference_clusters": int(
            pairwise["n_reference_clusters"].max()
        ),
        "n_batch_repeats": int(pairwise["n_repeats"].max()),
        "original_e12_repeated_seed_gate": original_gate,
        "mean_rmse_point_improves_vs_all_strong_baselines": (
            mean_point_supported
        ),
        "mean_rmse_intervals_confirm_vs_all_strong_baselines": (
            mean_interval_supported
        ),
        "mean_task_noninferiority_vs_all_strong_baselines": (
            mean_task_noninferiority
        ),
        "mean_material_harm_detected_vs_any_strong_baseline": (
            mean_harm_detected
        ),
        "mean_method_single_seed_strong_baseline_confirmed": bool(
            mean_interval_supported and mean_task_noninferiority
        ),
        "mean_method_ready_for_new_seed_confirmation": mean_ready,
        "all_classes_rmse_point_improve_vs_all_strong_baselines": (
            class_point_supported
        ),
        "all_classes_rmse_intervals_confirm_vs_all_strong_baselines": (
            class_interval_supported
        ),
        "class_material_harm_detected_vs_any_strong_baseline": (
            class_harm_detected
        ),
        "class_uniform_single_seed_strong_baseline_confirmed": bool(
            mean_interval_supported
            and class_interval_supported
            and mean_task_noninferiority
            and not class_harm_detected
        ),
        "class_uniform_method_ready_for_new_seed_confirmation": class_ready,
        "tightened_paper_level_class_uniform_claim_supported": False,
        "class_audit": [
            {key: _json_value(value) for key, value in row.items()}
            for row in class_audit.to_dict(orient="records")
        ],
        "decision_rule": (
            "Eligibility for new independent seeds is not confirmation: E12 "
            "must have lower point RMSE than both strict E11 and topology in "
            "random, balanced, and severe regimes, preserve BAcc and recall gap "
            "by paired noninferiority intervals, and show no material harm. "
            "Single-seed strong-baseline confirmation additionally requires "
            "every mean RMSE interval below zero. A class-uniform method must "
            "also improve every severe class against both strong baselines and "
            "show no class-specific harm; class-uniform confirmation additionally "
            "requires every class RMSE interval below zero. Because this gate "
            "was defined after reviewing E12 seed 7, it can reject a claim but "
            "cannot confirm one."
        ),
        "next_method_recommendation": (
            "run_new_probe_seeds_for_mean_only_claim_or_redesign_on_source_and_"
            "validate_class_uniform_claim_on_untouched_external_data"
            if mean_ready and not class_ready
            else "advance_frozen_method_to_new_probe_seeds_and_external_data"
            if class_ready
            else "do_not_run_multiseed_until_strong_baseline_failure_is_resolved"
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(output / "strong_baseline_pairwise.csv", index=False)
    class_audit.to_csv(output / "strong_baseline_class_audit.csv", index=False)
    with (output / "strong_baseline_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    return pairwise

"""E14 confirmation across untouched probe seeds under a frozen E13 gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .class_safeguard import E11_METHOD, SAFE_METHOD, TOPOLOGY_METHOD
from .prior_stress import _json_value
from .strong_baseline import (
    BACC_METRIC,
    GAP_METRIC,
    METRICS,
    RMSE_METRIC,
    STRONG_BASELINES,
    _comparison_specs,
    _decorate_comparison,
    _paired_cluster_matrices,
    _select_metrics,
    paired_cluster_bootstrap,
)

MEAN_COMPARISONS = {
    "primary_random",
    "balanced_stress_size",
    "severe_skew",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_frozen_gate(path: str | Path) -> dict[str, Any]:
    summary = _load_json(Path(path))
    if summary.get("stage") != "E13 post-hoc strongest-baseline audit":
        raise ValueError("The frozen gate must be an E13 summary")
    if summary.get("audit_status") != "post_hoc_falsification_only":
        raise ValueError("Seed-7 E13 must remain marked as post-hoc")
    if not summary.get("mean_method_ready_for_new_seed_confirmation"):
        raise ValueError("Frozen E13 did not license new mean-method seeds")
    if summary.get("class_uniform_method_ready_for_new_seed_confirmation"):
        raise ValueError("E14 expects the seed-7 class-uniform gate to be false")
    if tuple(summary.get("predeclared_strong_baselines", [])) != STRONG_BASELINES:
        raise ValueError("Frozen E13 strong baselines changed")
    return summary


def _validate_run_pair(
    logit_run: Path,
    e12_run: Path,
    *,
    exploratory_probe_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    logit_summary = _load_json(logit_run / "summary.json")
    e12_summary = _load_json(e12_run / "class_safeguard_summary.json")
    metrics_path = e12_run / "class_safeguard_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    required_logit_flags = {
        "validation_predictions_only": True,
        "prediction_split": "audit",
        "all_subject_splits_pairwise_disjoint": True,
        "probe_validation_audit_subjects_disjoint": True,
        "physionet_test_subjects_used_for_fitting_or_scoring": False,
        "encoder": "reve",
        "probe": "reve_set",
        "probe_objective": "car_only",
        "set_queries": 4,
        "set_heads": 8,
        "strict_determinism": True,
    }
    failed = [key for key, expected in required_logit_flags.items() if logit_summary.get(key) != expected]
    if failed:
        raise ValueError(f"Probe confirmation protocol failed: {failed}")

    probe_seed = int(logit_summary["probe_seed"])
    if probe_seed == exploratory_probe_seed:
        raise ValueError("The exploratory seed cannot enter new-seed confirmation")
    if int(logit_summary["reference_seed"]) != 7:
        raise ValueError("E14 freezes reference_seed=7")
    expected_subjects = {
        "train_subjects": list(range(1, 61)),
        "probe_validation_subjects": list(range(61, 71)),
        "audit_subjects": list(range(71, 90)),
        "reserved_test_subjects": list(range(90, 110)),
    }
    changed_splits = [
        key for key, expected in expected_subjects.items() if logit_summary.get(key) != expected
    ]
    if changed_splits:
        raise ValueError(f"Frozen E14 subject splits changed: {changed_splits}")
    if e12_summary.get("stage") != "E12 source-only class/operator trust safeguard":
        raise ValueError("Paired output is not an E12 run")
    if int(e12_summary.get("probe_seed", -1)) != probe_seed:
        raise ValueError("E12 probe seed does not match its logit run")
    if e12_summary.get("physionet_test_subjects_used"):
        raise ValueError("E12 must not use PhysioNet test subjects")
    e12_split_checks = [
        "source_adaptation_subjects_disjoint",
        "source_evaluation_subjects_disjoint",
        "adaptation_evaluation_subjects_disjoint",
    ]
    failed_e12_splits = [key for key in e12_split_checks if not e12_summary.get(key)]
    if failed_e12_splits:
        raise ValueError(f"E12 split checks failed: {failed_e12_splits}")

    audit_subjects = set(int(value) for value in logit_summary["audit_subjects"])
    e12_subjects = set(e12_summary["source_subjects"])
    e12_subjects |= set(e12_summary["adaptation_subjects"])
    e12_subjects |= set(e12_summary["evaluation_subjects"])
    if audit_subjects != e12_subjects:
        raise ValueError("E12 subjects do not exactly cover the frozen audit split")

    metrics = pd.read_csv(metrics_path)
    required_methods = {SAFE_METHOD, E11_METHOD, TOPOLOGY_METHOD}
    if not required_methods <= set(metrics["method"]):
        raise ValueError("E12 metrics are missing a strong-baseline method")
    return logit_summary, e12_summary, metrics


def _hierarchical_probe_bootstrap(
    candidate_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap crossed seed, repeat, and held-reference clusters."""

    candidate = np.asarray(candidate_values, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 3:
        raise ValueError("Expected matched seed x repeat x reference tensors")
    if candidate.shape[0] < 2:
        raise ValueError("E14 requires at least two untouched probe seeds")
    if n_resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap settings")

    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        seed_index = rng.integers(0, delta.shape[0], size=delta.shape[0])
        reference_index = rng.integers(0, delta.shape[2], size=delta.shape[2])
        seed_means = []
        for selected_seed in seed_index:
            repeat_index = rng.integers(0, delta.shape[1], size=delta.shape[1])
            seed_means.append(delta[selected_seed][np.ix_(repeat_index, reference_index)].mean())
        samples[index] = float(np.mean(seed_means))

    alpha = (1.0 - confidence) / 2.0
    return {
        "candidate_cluster_mean": float(candidate.mean()),
        "baseline_cluster_mean": float(baseline.mean()),
        "candidate_minus_baseline": float(delta.mean()),
        "ci_lower": float(np.quantile(samples, alpha)),
        "ci_upper": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "n_probe_seeds": int(delta.shape[0]),
        "n_repeats": int(delta.shape[1]),
        "n_reference_clusters": int(delta.shape[2]),
    }


def _class_summary(hierarchical: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    class_names = hierarchical.loc[
        hierarchical["dominant_class"].notna(),
        ["comparison", "dominant_class", "class_name"],
    ].drop_duplicates()
    for item in class_names.itertuples(index=False):
        selected = hierarchical.loc[hierarchical["comparison"] == item.comparison]
        rmse = selected.loc[selected["metric"] == RMSE_METRIC]
        tasks = selected.loc[selected["metric"].isin([BACC_METRIC, GAP_METRIC])]
        strongest = rmse.sort_values("baseline_cluster_mean").iloc[0]
        rows.append(
            {
                "comparison": item.comparison,
                "dominant_class": int(item.dominant_class),
                "class_name": item.class_name,
                "candidate_cluster_mean_rmse": float(strongest["candidate_cluster_mean"]),
                "strongest_baseline": strongest["baseline"],
                "strongest_baseline_cluster_mean_rmse": float(strongest["baseline_cluster_mean"]),
                "candidate_minus_strongest_rmse": float(strongest["candidate_minus_baseline"]),
                "strongest_ci_lower": float(strongest["ci_lower"]),
                "strongest_ci_upper": float(strongest["ci_upper"]),
                "rmse_point_improves_vs_both_strong_baselines": bool(rmse["point_improves"].all()),
                "rmse_intervals_confirm_vs_both_strong_baselines": bool(
                    rmse["interval_confirms_improvement"].all()
                ),
                "task_noninferiority_vs_both_strong_baselines": bool(
                    tasks["interval_confirms_noninferiority"].all()
                ),
                "material_harm_detected": bool(selected["material_harm_detected"].any()),
            }
        )
    return pd.DataFrame(rows)


def analyze_probe_seed_confirmation(
    frozen_e13_summary: str | Path,
    logit_runs: list[str | Path],
    e12_runs: list[str | Path],
    output_dir: str | Path,
    *,
    exploratory_probe_seed: int = 7,
    bootstrap_resamples: int = 10000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260720,
    max_bacc_loss: float = 0.01,
    max_gap_increase: float = 0.01,
) -> pd.DataFrame:
    """Confirm the frozen mean gate across untouched probe seeds."""

    if len(logit_runs) != len(e12_runs) or len(logit_runs) < 2:
        raise ValueError("Provide matched logit/E12 paths for at least two seeds")
    frozen = _validate_frozen_gate(frozen_e13_summary)
    if not np.isclose(float(frozen["max_bacc_loss"]), max_bacc_loss):
        raise ValueError("BAcc tolerance differs from frozen E13")
    if not np.isclose(float(frozen["max_gap_increase"]), max_gap_increase):
        raise ValueError("Recall-gap tolerance differs from frozen E13")

    run_data = []
    for logit_path, e12_path in zip(logit_runs, e12_runs, strict=True):
        logit_run = Path(logit_path)
        e12_run = Path(e12_path)
        logit_summary, e12_summary, metrics = _validate_run_pair(
            logit_run,
            e12_run,
            exploratory_probe_seed=exploratory_probe_seed,
        )
        run_data.append(
            {
                "probe_seed": int(logit_summary["probe_seed"]),
                "logit_run": logit_run,
                "e12_run": e12_run,
                "logit_summary": logit_summary,
                "e12_summary": e12_summary,
                "metrics": metrics,
            }
        )
    run_data.sort(key=lambda item: item["probe_seed"])
    probe_seeds = [item["probe_seed"] for item in run_data]
    if len(set(probe_seeds)) != len(probe_seeds):
        raise ValueError("Probe seeds must be unique")

    first_summary = run_data[0]["e12_summary"]
    specs = _comparison_specs(first_summary)
    per_seed_rows: list[dict[str, Any]] = []
    hierarchical_rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        selected_by_seed = [_select_metrics(item["metrics"], spec) for item in run_data]
        for baseline_index, baseline in enumerate(STRONG_BASELINES):
            for metric_index, metric in enumerate(METRICS):
                candidate_tensors = []
                baseline_tensors = []
                reference_grid: list[str] | None = None
                repeat_count: int | None = None
                for item, selected in zip(run_data, selected_by_seed, strict=True):
                    per_seed = paired_cluster_bootstrap(
                        selected,
                        candidate=SAFE_METHOD,
                        baseline=baseline,
                        metric=metric,
                        n_resamples=min(bootstrap_resamples, 5000),
                        confidence=bootstrap_confidence,
                        seed=(
                            bootstrap_seed
                            + 10000 * item["probe_seed"]
                            + 100 * spec_index
                            + 10 * baseline_index
                            + metric_index
                        ),
                    )
                    per_seed_rows.append(
                        _decorate_comparison(
                            {
                                "probe_seed": item["probe_seed"],
                                "comparison": spec.name,
                                "dominant_class": spec.dominant_class,
                                "class_name": spec.class_name,
                                **per_seed,
                            },
                            max_bacc_loss=max_bacc_loss,
                            max_gap_increase=max_gap_increase,
                        )
                    )
                    candidate, control, repeats, references, _ = _paired_cluster_matrices(
                        selected,
                        candidate=SAFE_METHOD,
                        baseline=baseline,
                        metric=metric,
                    )
                    if reference_grid is None:
                        reference_grid = references
                        repeat_count = len(repeats)
                    elif references != reference_grid or len(repeats) != repeat_count:
                        raise ValueError("Repeat/reference grids differ across probe seeds")
                    candidate_tensors.append(candidate)
                    baseline_tensors.append(control)

                hierarchical = _hierarchical_probe_bootstrap(
                    np.stack(candidate_tensors),
                    np.stack(baseline_tensors),
                    n_resamples=bootstrap_resamples,
                    confidence=bootstrap_confidence,
                    seed=(bootstrap_seed + 100 * spec_index + 10 * baseline_index + metric_index),
                )
                hierarchical_rows.append(
                    _decorate_comparison(
                        {
                            "comparison": spec.name,
                            "dominant_class": spec.dominant_class,
                            "class_name": spec.class_name,
                            "metric": metric,
                            "candidate": SAFE_METHOD,
                            "baseline": baseline,
                            **hierarchical,
                        },
                        max_bacc_loss=max_bacc_loss,
                        max_gap_increase=max_gap_increase,
                    )
                )

    per_seed = pd.DataFrame(per_seed_rows)
    hierarchical = pd.DataFrame(hierarchical_rows)
    class_summary = _class_summary(hierarchical)
    manifest = pd.DataFrame(
        [
            {
                "probe_seed": item["probe_seed"],
                "reference_seed": item["logit_summary"]["reference_seed"],
                "logit_run": str(item["logit_run"]),
                "e12_run": str(item["e12_run"]),
                "probe_validation_balanced_accuracy": item["logit_summary"]["validation_balanced_accuracy"],
                "selected_epoch": item["logit_summary"]["selected_epoch"],
                "all_subject_splits_pairwise_disjoint": item["logit_summary"][
                    "all_subject_splits_pairwise_disjoint"
                ],
                "physionet_test_subjects_used": item["e12_summary"]["physionet_test_subjects_used"],
                "original_e12_gate": item["e12_summary"][
                    "safeguard_supported_for_repeated_seed_confirmation"
                ],
            }
            for item in run_data
        ]
    )

    mean_hierarchical = hierarchical.loc[hierarchical["comparison"].isin(MEAN_COMPARISONS)]
    mean_rmse = mean_hierarchical.loc[mean_hierarchical["metric"] == RMSE_METRIC]
    mean_tasks = mean_hierarchical.loc[mean_hierarchical["metric"].isin([BACC_METRIC, GAP_METRIC])]
    per_seed_mean_rmse = per_seed.loc[
        per_seed["comparison"].isin(MEAN_COMPARISONS) & (per_seed["metric"] == RMSE_METRIC)
    ]
    per_seed_mean = (
        per_seed_mean_rmse.groupby("probe_seed")
        .agg(
            all_mean_rmse_points_improve=("point_improves", "all"),
            any_mean_material_harm=("material_harm_detected", "any"),
        )
        .reset_index()
    )
    direction_stable = bool(per_seed_mean["all_mean_rmse_points_improve"].all())
    hierarchical_rmse_confirmed = bool(mean_rmse["interval_confirms_improvement"].all())
    task_noninferiority = bool(mean_tasks["interval_confirms_noninferiority"].all())
    no_mean_harm = not bool(
        per_seed_mean["any_mean_material_harm"].any() or mean_hierarchical["material_harm_detected"].any()
    )
    confirmation_supported = bool(
        direction_stable and hierarchical_rmse_confirmed and task_noninferiority and no_mean_harm
    )

    class_uniform_new_seed_supported = bool(
        class_summary["rmse_point_improves_vs_both_strong_baselines"].all()
        and class_summary["rmse_intervals_confirm_vs_both_strong_baselines"].all()
        and class_summary["task_noninferiority_vs_both_strong_baselines"].all()
        and not class_summary["material_harm_detected"].any()
    )
    summary = {
        "stage": "E14 untouched-probe-seed mean-method confirmation",
        "gate_frozen_before_new_probe_results": True,
        "frozen_e13_summary": str(Path(frozen_e13_summary)),
        "exploratory_probe_seed_excluded_from_confirmation": (exploratory_probe_seed),
        "new_probe_seeds": probe_seeds,
        "n_new_probe_seeds": len(probe_seeds),
        "reference_seed": 7,
        "candidate": SAFE_METHOD,
        "strong_baselines": list(STRONG_BASELINES),
        "bootstrap_unit": ("hierarchical crossed probe seed x batch repeat x held-reference identity"),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_confidence": bootstrap_confidence,
        "bootstrap_seed": bootstrap_seed,
        "max_bacc_loss": max_bacc_loss,
        "max_gap_increase": max_gap_increase,
        "probe_train_validation_audit_test_pairwise_disjoint": bool(
            manifest["all_subject_splits_pairwise_disjoint"].all()
        ),
        "physionet_test_subjects_used": bool(manifest["physionet_test_subjects_used"].any()),
        "every_new_seed_mean_rmse_point_improves_vs_both_strong_baselines": (direction_stable),
        "hierarchical_mean_rmse_intervals_confirm_vs_both_strong_baselines": (hierarchical_rmse_confirmed),
        "hierarchical_mean_task_noninferiority": task_noninferiority,
        "mean_material_harm_detected": not no_mean_harm,
        "mean_method_new_seed_confirmation_supported": confirmation_supported,
        "new_seed_class_uniform_diagnostic_supported": (class_uniform_new_seed_supported),
        "current_class_uniform_claim_remains_rejected": True,
        "paper_level_mean_claim_supported": False,
        "paper_claim_blockers": [
            "single_dataset",
            "only_two_untouched_probe_seeds" if len(probe_seeds) < 3 else "external_dataset_not_run",
            "E13_seed7_gate_was_post_hoc",
        ],
        "per_seed_mean_gate": [
            {key: _json_value(value) for key, value in row.items()}
            for row in per_seed_mean.to_dict(orient="records")
        ],
        "class_diagnostic": [
            {key: _json_value(value) for key, value in row.items()}
            for row in class_summary.to_dict(orient="records")
        ],
        "decision_rule": (
            "Using only untouched probe seeds, require lower mean RMSE point "
            "estimates against strict E11 and topology in every seed and all "
            "three frozen regimes; require crossed hierarchical RMSE intervals "
            "below zero, BAcc/recall-gap noninferiority, and no material mean "
            "harm. Seed 7 is excluded from confirmation. Class-wise results "
            "remain a falsification diagnostic and cannot revive the current "
            "class-uniform claim."
        ),
        "next_method_recommendation": (
            "validate_frozen_mean_method_on_external_open_eeg_dataset"
            if confirmation_supported
            else "do_not_tune_on_audit_subjects_revisit_source_only_method"
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "probe_seed_manifest.csv", index=False)
    per_seed.to_csv(output / "probe_seed_pairwise.csv", index=False)
    hierarchical.to_csv(output / "probe_seed_hierarchical_pairwise.csv", index=False)
    class_summary.to_csv(output / "probe_seed_class_diagnostic.csv", index=False)
    with (output / "probe_seed_confirmation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return hierarchical

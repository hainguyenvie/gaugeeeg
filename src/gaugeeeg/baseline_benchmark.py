"""Frozen-REVE baseline sweep for the locked GaugeEEG development benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

EVALUATION_VIEWS = (
    "car",
    "cz",
    "pz",
    "fz",
    "native32@car",
    "native32@cz",
    "native32@pz",
    "native32@fz",
    "native16@car",
    "native16@cz",
    "native16@pz",
    "native16@fz",
)

JOINT_TRAINING_VIEWS = (
    "car",
    "pz",
    "fz",
    "native32@car",
    "native32@pz",
    "native32@fz",
    "native16@car",
    "native16@pz",
    "native16@fz",
)


@dataclass(frozen=True)
class BaselineSpec:
    training_views: tuple[str, ...]
    objective: str
    consistency_weight: float = 0.0


BASELINE_SPECS: dict[str, BaselineSpec] = {
    "car_only": BaselineSpec(("car",), "car_only"),
    # Cz remains held out so this baseline measures reference generalization.
    "reference_multiview_ce": BaselineSpec(("car", "pz", "fz"), "multi_view_ce"),
    "structured_montage_ce": BaselineSpec(("car", "native32@car", "native16@car"), "multi_view_ce"),
    "joint_multiview_ce": BaselineSpec(JOINT_TRAINING_VIEWS, "multi_view_ce"),
    # Three deterministic random nested 64->32->16 layouts approximate random
    # channel-dropout augmentation without zero padding.
    "random_montage_ce": BaselineSpec(
        (
            "car",
            "native_random32_s101@car",
            "native_random16_s101@car",
            "native_random32_s202@car",
            "native_random16_s202@car",
            "native_random32_s303@car",
            "native_random16_s303@car",
        ),
        "multi_view_ce",
    ),
    "region_dropout_ce": BaselineSpec(
        ("car", "native_drop_left_motor@car", "native_drop_right_motor@car"),
        "multi_view_ce",
    ),
    # Generic generalized Jensen-Shannon consistency is deliberately not
    # presented as a GaugeEEG rule contribution.
    "joint_js_consistency": BaselineSpec(JOINT_TRAINING_VIEWS, "rule_consistency", consistency_weight=1.0),
}

EXPECTED_TRAIN = tuple(range(1, 61))
EXPECTED_VALIDATION = tuple(range(61, 71))
EXPECTED_AUDIT = tuple(range(71, 90))
HISTORICALLY_INSPECTED_TEST = tuple(range(90, 110))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_run_specs(run_specs: list[str]) -> dict[str, list[Path]]:
    grouped = {method: [] for method in BASELINE_SPECS}
    for item in run_specs:
        if "=" not in item:
            raise ValueError(f"Invalid run specification {item!r}; expected method=path")
        method, raw_path = item.split("=", maxsplit=1)
        if method not in BASELINE_SPECS:
            raise ValueError(f"Unknown baseline {method!r}; expected {sorted(BASELINE_SPECS)}")
        grouped[method].append(Path(raw_path))
    missing = [method for method, paths in grouped.items() if not paths]
    if missing:
        raise ValueError(f"Missing required baseline runs: {missing}")
    return grouped


def _normalize_views(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(value).casefold() for value in values)


def _validate_run(
    method: str,
    run_dir: Path,
    *,
    spec: BaselineSpec | None = None,
    expected_defense: str = "none",
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "validation_predictions.csv"
    if not summary_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(f"Incomplete baseline run: {run_dir}")
    summary = _load_json(summary_path)
    spec = BASELINE_SPECS[method] if spec is None else spec
    checks = {
        "validation_only": summary.get("validation_predictions_only") is True,
        "prediction_split": summary.get("prediction_split") == "audit",
        "test_not_scored": summary.get("physionet_test_subjects_used_for_fitting_or_scoring") is False,
        "pairwise_disjoint": summary.get("all_subject_splits_pairwise_disjoint") is True,
        "q4": summary.get("set_queries") == 4,
        "objective": summary.get("probe_objective") == spec.objective,
        "training_views": _normalize_views(summary.get("training_views", []))
        == _normalize_views(spec.training_views),
        "evaluation_views": set(_normalize_views(summary.get("validation_prediction_views", [])))
        == set(EVALUATION_VIEWS),
        "consistency_weight": np.isclose(
            float(summary.get("consistency_weight", 0.0)), spec.consistency_weight
        ),
        "train_split": tuple(summary.get("train_subjects", [])) == EXPECTED_TRAIN,
        "probe_validation_split": tuple(summary.get("probe_validation_subjects", [])) == EXPECTED_VALIDATION,
        "audit_split": tuple(summary.get("audit_subjects", [])) == EXPECTED_AUDIT,
        "historical_test_field": tuple(summary.get("reserved_test_subjects", []))
        == HISTORICALLY_INSPECTED_TEST,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Baseline run {run_dir} failed locked checks: {failed}")

    predictions = pd.read_csv(predictions_path)
    required = {"defense", "test_view", "trial_index", "subject_id", "y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns in {predictions_path}: {missing}")
    predictions["test_view"] = predictions["test_view"].str.casefold()
    observed_defenses = set(predictions["defense"].astype(str).str.casefold())
    if observed_defenses != {expected_defense.casefold()}:
        raise ValueError(f"Unexpected defenses in {predictions_path}: {sorted(observed_defenses)}")
    if set(predictions["test_view"]) != set(EVALUATION_VIEWS):
        raise ValueError(f"Unexpected evaluation grid in {predictions_path}")

    by_view: dict[str, pd.DataFrame] = {}
    alignment: pd.DataFrame | None = None
    for view in EVALUATION_VIEWS:
        frame = predictions.loc[predictions["test_view"] == view].sort_values(["trial_index"])
        key = frame[["trial_index", "subject_id", "y_true"]].reset_index(drop=True)
        if alignment is None:
            alignment = key
        elif not alignment.equals(key):
            raise ValueError(f"Trial alignment changed across views in {predictions_path}")
        by_view[view] = frame.reset_index(drop=True)
    if alignment is None or set(alignment["subject_id"]) != set(EXPECTED_AUDIT):
        raise ValueError(f"Audit subjects are incomplete in {predictions_path}")
    return summary, by_view


def _metric_row(
    method: str,
    run_dir: Path,
    probe_seed: int,
    view: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    labels = frame["y_true"].to_numpy(dtype=np.int64)
    prediction = frame["y_pred"].to_numpy(dtype=np.int64)
    recalls = recall_score(labels, prediction, labels=np.arange(4), average=None, zero_division=0)
    return {
        "method": method,
        "run_dir": str(run_dir),
        "probe_seed": probe_seed,
        "test_view": view,
        "n_trials": int(labels.size),
        "n_subjects": int(frame["subject_id"].nunique()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "worst_class_recall": float(recalls.min()),
        **{f"class_{index}_recall": float(value) for index, value in enumerate(recalls)},
    }


def _hierarchical_bacc_delta(
    candidate: dict[int, pd.DataFrame],
    baseline: dict[int, pd.DataFrame],
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    probe_seeds = sorted(candidate)
    if probe_seeds != sorted(baseline):
        raise ValueError("Candidate and baseline probe seeds differ")
    point_deltas = []
    for probe_seed in probe_seeds:
        candidate_frame = candidate[probe_seed]
        baseline_frame = baseline[probe_seed]
        key = ["trial_index", "subject_id", "y_true"]
        if not candidate_frame[key].equals(baseline_frame[key]):
            raise ValueError("Candidate and baseline predictions are not paired")
        labels = candidate_frame["y_true"].to_numpy(dtype=np.int64)
        point_deltas.append(
            balanced_accuracy_score(labels, candidate_frame["y_pred"])
            - balanced_accuracy_score(labels, baseline_frame["y_pred"])
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for draw_index in range(n_resamples):
        sampled_seeds = rng.choice(probe_seeds, size=len(probe_seeds), replace=True)
        seed_deltas = []
        for probe_seed in sampled_seeds:
            candidate_frame = candidate[int(probe_seed)]
            baseline_frame = baseline[int(probe_seed)]
            subjects = candidate_frame["subject_id"].unique()
            sampled_subjects = rng.choice(subjects, size=subjects.size, replace=True)
            positions = np.concatenate(
                [
                    np.flatnonzero(candidate_frame["subject_id"].to_numpy() == subject)
                    for subject in sampled_subjects
                ]
            )
            labels = candidate_frame["y_true"].to_numpy(dtype=np.int64)[positions]
            seed_deltas.append(
                balanced_accuracy_score(labels, candidate_frame["y_pred"].to_numpy(dtype=np.int64)[positions])
                - balanced_accuracy_score(
                    labels, baseline_frame["y_pred"].to_numpy(dtype=np.int64)[positions]
                )
            )
        draws[draw_index] = float(np.mean(seed_deltas))
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_probe_seeds": len(probe_seeds),
        "n_resamples": n_resamples,
        "confidence": confidence,
        "candidate_minus_baseline": float(np.mean(point_deltas)),
        "bootstrap_mean": float(draws.mean()),
        "ci_lower": float(np.quantile(draws, alpha)),
        "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
    }


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = [
        float(np.mean(predictions[labels == class_index] == class_index)) for class_index in np.unique(labels)
    ]
    return float(np.mean(recalls))


def _hierarchical_bacc_delta_views(
    candidate: dict[int, dict[str, pd.DataFrame]],
    baseline: dict[int, dict[str, pd.DataFrame]],
    *,
    views: tuple[str, ...],
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap the mean view-level BAcc delta with paired seed/subject draws."""

    probe_seeds = sorted(candidate)
    if probe_seeds != sorted(baseline):
        raise ValueError("Candidate and baseline probe seeds differ")
    keys = ["trial_index", "subject_id", "y_true"]
    point_deltas: list[float] = []
    for probe_seed in probe_seeds:
        view_deltas = []
        for view in views:
            candidate_frame = candidate[probe_seed][view]
            baseline_frame = baseline[probe_seed][view]
            if not candidate_frame[keys].equals(baseline_frame[keys]):
                raise ValueError("Candidate and baseline predictions are not paired")
            labels = candidate_frame["y_true"].to_numpy(dtype=np.int64)
            view_deltas.append(
                _balanced_accuracy(labels, candidate_frame["y_pred"].to_numpy(dtype=np.int64))
                - _balanced_accuracy(labels, baseline_frame["y_pred"].to_numpy(dtype=np.int64))
            )
        point_deltas.append(float(np.mean(view_deltas)))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for draw_index in range(n_resamples):
        sampled_seeds = rng.choice(probe_seeds, size=len(probe_seeds), replace=True)
        seed_deltas = []
        for sampled_seed in sampled_seeds:
            probe_seed = int(sampled_seed)
            first_frame = candidate[probe_seed][views[0]]
            subjects = first_frame["subject_id"].unique()
            sampled_subjects = rng.choice(subjects, size=subjects.size, replace=True)
            positions = np.concatenate(
                [
                    np.flatnonzero(first_frame["subject_id"].to_numpy() == subject)
                    for subject in sampled_subjects
                ]
            )
            view_deltas = []
            for view in views:
                candidate_frame = candidate[probe_seed][view]
                baseline_frame = baseline[probe_seed][view]
                labels = candidate_frame["y_true"].to_numpy(dtype=np.int64)[positions]
                view_deltas.append(
                    _balanced_accuracy(
                        labels,
                        candidate_frame["y_pred"].to_numpy(dtype=np.int64)[positions],
                    )
                    - _balanced_accuracy(
                        labels,
                        baseline_frame["y_pred"].to_numpy(dtype=np.int64)[positions],
                    )
                )
            seed_deltas.append(float(np.mean(view_deltas)))
        draws[draw_index] = float(np.mean(seed_deltas))
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_probe_seeds": len(probe_seeds),
        "n_resamples": n_resamples,
        "confidence": confidence,
        "candidate_minus_baseline": float(np.mean(point_deltas)),
        "bootstrap_mean": float(draws.mean()),
        "ci_lower": float(np.quantile(draws, alpha)),
        "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
    }


def analyze_baseline_benchmark(
    run_specs: list[str],
    output_dir: str | Path,
    *,
    expected_seeds: list[int] | tuple[int, ...] = (7, 21, 42),
    bootstrap_resamples: int = 10_000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260720,
) -> pd.DataFrame:
    """Validate, aggregate, and rank a complete locked baseline matrix."""

    grouped = _parse_run_specs(run_specs)
    expected_seed_set = {int(value) for value in expected_seeds}
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    frames: dict[str, dict[int, dict[str, pd.DataFrame]]] = {}
    encoder_revisions: set[tuple[str, str]] = set()
    dataset_fingerprints: set[str] = set()

    for method, run_dirs in grouped.items():
        frames[method] = {}
        observed_seeds: set[int] = set()
        for run_dir in run_dirs:
            summary, by_view = _validate_run(method, run_dir)
            probe_seed = int(summary["probe_seed"])
            if probe_seed in observed_seeds:
                raise ValueError(f"Duplicate probe seed {probe_seed} for {method}")
            observed_seeds.add(probe_seed)
            frames[method][probe_seed] = by_view
            metadata = summary.get("encoder_metadata", {})
            encoder_revisions.add(
                (str(metadata.get("model_revision")), str(metadata.get("position_model_revision")))
            )
            dataset_fingerprints.add(str(summary.get("dataset_fingerprint")))
            manifest_rows.append(
                {
                    "method": method,
                    "run_dir": str(run_dir),
                    "probe_seed": probe_seed,
                    "objective": summary["probe_objective"],
                    "training_views": "|".join(_normalize_views(summary["training_views"])),
                    "dataset_fingerprint": summary.get("dataset_fingerprint"),
                    "model_revision": metadata.get("model_revision"),
                    "position_model_revision": metadata.get("position_model_revision"),
                    "historical_test_scored": False,
                }
            )
            rows.extend(
                _metric_row(method, run_dir, probe_seed, view, by_view[view]) for view in EVALUATION_VIEWS
            )
        if observed_seeds != expected_seed_set:
            raise ValueError(
                f"{method} has probe seeds {sorted(observed_seeds)}, expected {sorted(expected_seed_set)}"
            )

    if len(encoder_revisions) != 1:
        raise ValueError(f"Baseline runs use different REVE revisions: {encoder_revisions}")
    only_revisions = next(iter(encoder_revisions))
    invalid_revisions = {"", "None", "unresolved"}
    if any(revision in invalid_revisions for revision in only_revisions):
        raise ValueError(f"Baseline runs do not resolve immutable REVE revisions: {only_revisions}")
    if len(dataset_fingerprints) != 1 or "None" in dataset_fingerprints:
        raise ValueError(f"Baseline runs use invalid dataset fingerprints: {dataset_fingerprints}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_seed = pd.DataFrame(rows).sort_values(["method", "probe_seed", "test_view"])
    by_seed.to_csv(output / "baseline_metrics_by_seed.csv", index=False)
    manifest = pd.DataFrame(manifest_rows).sort_values(["method", "probe_seed"])
    manifest.to_csv(output / "baseline_manifest.csv", index=False)

    aggregate = (
        by_seed.groupby(["method", "test_view"], as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            worst_class_recall_mean=("worst_class_recall", "mean"),
        )
        .sort_values(["method", "test_view"])
    )
    aggregate.to_csv(output / "baseline_metrics_summary.csv", index=False)

    bootstrap_rows: list[dict[str, Any]] = []
    primary_views = ("car", "native32@car", "native16@car", "native16@cz")
    for candidate in BASELINE_SPECS:
        for baseline in ("car_only", "joint_multiview_ce"):
            if candidate == baseline:
                continue
            for view_index, view in enumerate(primary_views):
                result = _hierarchical_bacc_delta(
                    {seed: frames[candidate][seed][view] for seed in expected_seed_set},
                    {seed: frames[baseline][seed][view] for seed in expected_seed_set},
                    n_resamples=bootstrap_resamples,
                    confidence=bootstrap_confidence,
                    seed=bootstrap_seed + view_index,
                )
                bootstrap_rows.append(
                    {"candidate": candidate, "baseline": baseline, "test_view": view, **result}
                )
            native16_views = tuple(view for view in EVALUATION_VIEWS if view.startswith("native16@"))
            result = _hierarchical_bacc_delta_views(
                frames[candidate],
                frames[baseline],
                views=native16_views,
                n_resamples=bootstrap_resamples,
                confidence=bootstrap_confidence,
                seed=bootstrap_seed + len(primary_views),
            )
            bootstrap_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "test_view": "native16_reference_mean",
                    **result,
                }
            )
    pairwise = pd.DataFrame(bootstrap_rows)
    pairwise.to_csv(output / "baseline_pairwise_bootstrap.csv", index=False)

    car_clean = float(
        aggregate.loc[
            (aggregate["method"] == "car_only") & (aggregate["test_view"] == "car"),
            "balanced_accuracy_mean",
        ].iloc[0]
    )
    method_rows: list[dict[str, Any]] = []
    native16_views = [view for view in EVALUATION_VIEWS if view.startswith("native16@")]
    for method in BASELINE_SPECS:
        selected = aggregate.loc[aggregate["method"] == method].set_index("test_view")
        clean = float(selected.loc["car", "balanced_accuracy_mean"])
        if method == "car_only":
            clean_ci_lower = 0.0
            clean_ci_upper = 0.0
        else:
            clean_comparison = pairwise.loc[
                (pairwise["candidate"] == method)
                & (pairwise["baseline"] == "car_only")
                & (pairwise["test_view"] == "car")
            ].iloc[0]
            clean_ci_lower = float(clean_comparison["ci_lower"])
            clean_ci_upper = float(clean_comparison["ci_upper"])
        clean_delta = clean - car_clean
        method_rows.append(
            {
                "method": method,
                "clean_car_bacc": clean,
                "clean_delta_vs_car_only": clean_delta,
                "clean_delta_ci_lower": clean_ci_lower,
                "clean_delta_ci_upper": clean_ci_upper,
                "clean_noninferiority_passed": bool(clean_delta >= -0.01 and clean_ci_lower >= -0.01),
                "native32_car_bacc": float(selected.loc["native32@car", "balanced_accuracy_mean"]),
                "native16_car_bacc": float(selected.loc["native16@car", "balanced_accuracy_mean"]),
                "full_reference_bacc_mean": float(
                    selected.loc[["car", "cz", "pz", "fz"], "balanced_accuracy_mean"].mean()
                ),
                "native16_reference_bacc_mean": float(
                    selected.loc[native16_views, "balanced_accuracy_mean"].mean()
                ),
                "native16_worst_class_recall": float(
                    selected.loc[native16_views, "worst_class_recall_mean"].min()
                ),
                "suite_bacc_mean": float(selected["balanced_accuracy_mean"].mean()),
                "suite_worst_view_bacc": float(selected["balanced_accuracy_mean"].min()),
            }
        )
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(output / "baseline_method_summary.csv", index=False)
    eligible = method_summary.loc[method_summary["clean_noninferiority_passed"]].sort_values(
        ["native16_reference_bacc_mean", "suite_bacc_mean"], ascending=False
    )
    strongest = str(eligible.iloc[0]["method"]) if not eligible.empty else None

    summary = {
        "stage": "GaugeEEG frozen-REVE development benchmark lock",
        "status": "development_baseline_selection_only",
        "physionetmi_is_globally_untouched_test": False,
        "historically_inspected_subjects": list(HISTORICALLY_INSPECTED_TEST),
        "external_dataset_required_for_confirmation": True,
        "expected_probe_seeds": sorted(expected_seed_set),
        "evaluation_views": list(EVALUATION_VIEWS),
        "required_baselines": list(BASELINE_SPECS),
        "primary_selection_metric": "mean BAcc over native16@{CAR,Cz,Pz,Fz}",
        "clean_noninferiority_margin": 0.01,
        "strongest_development_baseline": strongest,
        "selection_is_paper_confirmation": False,
        "dataset_fingerprint": next(iter(dataset_fingerprints)),
        "encoder_revisions": list(only_revisions),
        "bootstrap_unit": "probe seed x subject",
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_confidence": bootstrap_confidence,
        "output_files": {
            "manifest": str(output / "baseline_manifest.csv"),
            "by_seed": str(output / "baseline_metrics_by_seed.csv"),
            "aggregate": str(output / "baseline_metrics_summary.csv"),
            "method_summary": str(output / "baseline_method_summary.csv"),
            "pairwise_bootstrap": str(output / "baseline_pairwise_bootstrap.csv"),
        },
    }
    with (output / "benchmark_lock_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return method_summary

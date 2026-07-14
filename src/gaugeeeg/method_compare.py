"""Predeclared comparisons for the held-out-reference consistency screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


PRIMARY_REDUCTION_THRESHOLD = 0.30
MAXIMUM_CLEAN_BACC_DROP = 0.01


def _aligned_predictions(
    run_dir: str | Path, *, target_view: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_path = Path(run_dir) / "predictions.csv"
    predictions = pd.read_csv(prediction_path)
    required = {"test_view", "trial_index", "subject_id", "y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{prediction_path} is missing columns: {missing}")

    car = predictions[predictions["test_view"].str.casefold() == "car"].sort_values(
        ["subject_id", "trial_index"]
    )
    shifted = predictions[
        predictions["test_view"].str.casefold() == target_view.casefold()
    ].sort_values(["subject_id", "trial_index"])
    identifiers = ["subject_id", "trial_index", "y_true"]
    if car.empty or shifted.empty:
        raise ValueError(f"{prediction_path} must contain CAR and {target_view} predictions")
    if not np.array_equal(car[identifiers].to_numpy(), shifted[identifiers].to_numpy()):
        raise RuntimeError(f"Prediction trials are not aligned in {prediction_path}")
    return car.reset_index(drop=True), shifted.reset_index(drop=True)


def _run_seed(run_dir: str | Path) -> int:
    metrics = pd.read_csv(Path(run_dir) / "metrics.csv")
    if "probe_seed" not in metrics:
        predictions = pd.read_csv(Path(run_dir) / "predictions.csv", usecols=["probe_seed"])
        values = predictions["probe_seed"].unique()
    else:
        values = metrics["probe_seed"].unique()
    if len(values) != 1:
        raise ValueError(f"Expected one probe seed in {run_dir}, found {values.tolist()}")
    return int(values[0])


def _recall_gap(
    y_true: np.ndarray,
    car_prediction: np.ndarray,
    shifted_prediction: np.ndarray,
    class_index: int,
) -> float:
    class_mask = y_true == class_index
    if not np.any(class_mask):
        raise ValueError(f"Target class {class_index} is absent from the sampled predictions")
    return float(
        np.mean(car_prediction[class_mask] == class_index)
        - np.mean(shifted_prediction[class_mask] == class_index)
    )


def _balanced_accuracy(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, prediction))


def _run_effect(run_dir: str | Path, *, target_view: str, target_class: int) -> dict[str, float]:
    run_path = Path(run_dir)
    metrics = pd.read_csv(run_path / "metrics.csv")
    car_metric = metrics[metrics["test_view"].str.casefold() == "car"].iloc[0]
    shifted_metric = metrics[metrics["test_view"].str.casefold() == target_view.casefold()].iloc[0]
    car, shifted = _aligned_predictions(run_path, target_view=target_view)
    y_true = car["y_true"].to_numpy(dtype=np.int64)
    recall_gap = _recall_gap(
        y_true,
        car["y_pred"].to_numpy(dtype=np.int64),
        shifted["y_pred"].to_numpy(dtype=np.int64),
        target_class,
    )
    return {
        "car_balanced_accuracy": float(car_metric["balanced_accuracy"]),
        "target_balanced_accuracy": float(shifted_metric["balanced_accuracy"]),
        "balanced_accuracy_gap": float(
            car_metric["balanced_accuracy"] - shifted_metric["balanced_accuracy"]
        ),
        "target_class_recall_gap": recall_gap,
    }


def _comparison_frame(
    baseline_dir: str | Path,
    augmentation_dir: str | Path,
    consistency_dir: str | Path,
    *,
    target_view: str,
    target_class: int,
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
    return comparison


def _paired_arrays(
    augmentation_dir: str | Path,
    consistency_dir: str | Path,
    *,
    target_view: str,
) -> dict[str, np.ndarray]:
    aug_car, aug_target = _aligned_predictions(augmentation_dir, target_view=target_view)
    con_car, con_target = _aligned_predictions(consistency_dir, target_view=target_view)
    identifiers = ["subject_id", "trial_index", "y_true"]
    if not np.array_equal(aug_car[identifiers].to_numpy(), con_car[identifiers].to_numpy()):
        raise RuntimeError(
            f"Trials differ between augmentation={augmentation_dir} and consistency={consistency_dir}"
        )
    return {
        "subjects": aug_car["subject_id"].to_numpy(dtype=np.int64),
        "y_true": aug_car["y_true"].to_numpy(dtype=np.int64),
        "aug_car": aug_car["y_pred"].to_numpy(dtype=np.int64),
        "aug_target": aug_target["y_pred"].to_numpy(dtype=np.int64),
        "con_car": con_car["y_pred"].to_numpy(dtype=np.int64),
        "con_target": con_target["y_pred"].to_numpy(dtype=np.int64),
    }


def _improvement_metrics(
    arrays: dict[str, np.ndarray], indices: np.ndarray, target_class: int
) -> dict[str, float]:
    y_true = arrays["y_true"][indices]
    aug_car = arrays["aug_car"][indices]
    aug_target = arrays["aug_target"][indices]
    con_car = arrays["con_car"][indices]
    con_target = arrays["con_target"][indices]

    aug_recall_gap = _recall_gap(y_true, aug_car, aug_target, target_class)
    con_recall_gap = _recall_gap(y_true, con_car, con_target, target_class)
    aug_car_bacc = _balanced_accuracy(y_true, aug_car)
    aug_target_bacc = _balanced_accuracy(y_true, aug_target)
    con_car_bacc = _balanced_accuracy(y_true, con_car)
    con_target_bacc = _balanced_accuracy(y_true, con_target)
    return {
        "target_class_recall_gap_recovery": aug_recall_gap - con_recall_gap,
        "balanced_accuracy_gap_recovery": (
            aug_car_bacc - aug_target_bacc
        ) - (con_car_bacc - con_target_bacc),
        "clean_balanced_accuracy_gain": con_car_bacc - aug_car_bacc,
        "target_balanced_accuracy_gain": con_target_bacc - aug_target_bacc,
    }


def _summarize_bootstrap(
    point: dict[str, float],
    samples: dict[str, np.ndarray],
    *,
    confidence: float,
    n_subjects: int,
    n_probe_seeds: int,
) -> pd.DataFrame:
    alpha = (1.0 - confidence) / 2.0
    rows = []
    for metric, values in samples.items():
        lower = float(np.quantile(values, alpha))
        upper = float(np.quantile(values, 1.0 - alpha))
        rows.append(
            {
                "metric": metric,
                "positive_means": "rule_consistency_better_than_multi_view_ce",
                "point_estimate": float(point[metric]),
                "bootstrap_mean": float(values.mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "probability_improvement_positive": float(np.mean(values > 0.0)),
                "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                "n_subjects": int(n_subjects),
                "n_probe_seeds": int(n_probe_seeds),
                "n_resamples": int(values.size),
                "confidence": float(confidence),
            }
        )
    return pd.DataFrame(rows)


def paired_method_bootstrap(
    augmentation_dir: str | Path,
    consistency_dir: str | Path,
    *,
    target_view: str = "cz",
    target_class: int = 0,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260714,
) -> pd.DataFrame:
    """Bootstrap the direct method difference by resampling test subjects."""

    arrays = _paired_arrays(augmentation_dir, consistency_dir, target_view=target_view)
    unique_subjects = np.unique(arrays["subjects"])
    subject_indices = {
        subject: np.flatnonzero(arrays["subjects"] == subject) for subject in unique_subjects
    }
    point_indices = np.arange(arrays["y_true"].size)
    point = _improvement_metrics(arrays, point_indices, target_class)
    samples = {metric: np.empty(n_resamples, dtype=np.float64) for metric in point}
    rng = np.random.default_rng(seed)
    for bootstrap_index in range(n_resamples):
        sampled_subjects = rng.choice(unique_subjects, size=unique_subjects.size, replace=True)
        indices = np.concatenate([subject_indices[subject] for subject in sampled_subjects])
        values = _improvement_metrics(arrays, indices, target_class)
        for metric, value in values.items():
            samples[metric][bootstrap_index] = value
    return _summarize_bootstrap(
        point,
        samples,
        confidence=confidence,
        n_subjects=unique_subjects.size,
        n_probe_seeds=1,
    )


def compare_consistency_methods(
    baseline_dir: str | Path,
    augmentation_dir: str | Path,
    consistency_dir: str | Path,
    output_dir: str | Path,
    *,
    target_view: str = "cz",
    target_class: int = 0,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20260714,
) -> pd.DataFrame:
    comparison = _comparison_frame(
        baseline_dir,
        augmentation_dir,
        consistency_dir,
        target_view=target_view,
        target_class=target_class,
    )
    paired = paired_method_bootstrap(
        augmentation_dir,
        consistency_dir,
        target_view=target_view,
        target_class=target_class,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=bootstrap_seed,
    )

    augmentation = comparison[comparison["method"] == "multi_view_ce"].iloc[0]
    consistency = comparison[comparison["method"] == "rule_consistency"].iloc[0]
    primary_bootstrap = paired[
        paired["metric"] == "target_class_recall_gap_recovery"
    ].iloc[0]
    summary = {
        "target_view": target_view,
        "target_class_index": int(target_class),
        "predeclared_primary_metric": "held-out-view target-class recall gap",
        "success_thresholds": {
            "relative_recall_gap_reduction_vs_car_only": PRIMARY_REDUCTION_THRESHOLD,
            "maximum_clean_balanced_accuracy_drop": MAXIMUM_CLEAN_BACC_DROP,
        },
        "augmentation_passes": bool(
            augmentation["target_class_recall_gap_relative_reduction"]
            >= PRIMARY_REDUCTION_THRESHOLD
            and augmentation["clean_balanced_accuracy_change"] >= -MAXIMUM_CLEAN_BACC_DROP
        ),
        "consistency_passes": bool(
            consistency["target_class_recall_gap_relative_reduction"]
            >= PRIMARY_REDUCTION_THRESHOLD
            and consistency["clean_balanced_accuracy_change"] >= -MAXIMUM_CLEAN_BACC_DROP
        ),
        "consistency_beats_augmentation_on_primary": bool(
            consistency["target_class_recall_gap"] < augmentation["target_class_recall_gap"]
        ),
        "paired_primary_improvement": {
            key: primary_bootstrap[key].item()
            if isinstance(primary_bootstrap[key], np.generic)
            else primary_bootstrap[key]
            for key in (
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "probability_improvement_positive",
                "ci_excludes_zero",
            )
        },
        "interpretation_rule": (
            "A seed-level direction is exploratory. Claim rule-loss value only after the "
            "multi-seed hierarchical bootstrap excludes zero in favor of consistency."
        ),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path / "method_comparison.csv", index=False)
    paired.to_csv(output_path / "paired_method_bootstrap.csv", index=False)
    with (output_path / "method_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return comparison


def aggregate_consistency_methods(
    baseline_dirs: list[str | Path],
    augmentation_dirs: list[str | Path],
    consistency_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    target_view: str = "cz",
    target_class: int = 0,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20260714,
) -> pd.DataFrame:
    """Aggregate method effects and run a seed-and-subject hierarchical bootstrap."""

    lengths = {len(baseline_dirs), len(augmentation_dirs), len(consistency_dirs)}
    if len(lengths) != 1 or not baseline_dirs:
        raise ValueError("Baseline, augmentation, and consistency run lists must be non-empty and equal")

    by_seed_frames = []
    paired_runs: list[tuple[int, dict[str, np.ndarray]]] = []
    for baseline_dir, augmentation_dir, consistency_dir in zip(
        baseline_dirs, augmentation_dirs, consistency_dirs, strict=True
    ):
        seeds = {
            _run_seed(baseline_dir),
            _run_seed(augmentation_dir),
            _run_seed(consistency_dir),
        }
        if len(seeds) != 1:
            raise ValueError(
                "Each baseline/augmentation/consistency triple must share one probe seed: "
                f"{baseline_dir}, {augmentation_dir}, {consistency_dir}"
            )
        probe_seed = seeds.pop()
        frame = _comparison_frame(
            baseline_dir,
            augmentation_dir,
            consistency_dir,
            target_view=target_view,
            target_class=target_class,
        )
        frame.insert(0, "probe_seed", probe_seed)
        by_seed_frames.append(frame)
        paired_runs.append(
            (
                probe_seed,
                _paired_arrays(augmentation_dir, consistency_dir, target_view=target_view),
            )
        )

    seed_values = [seed_value for seed_value, _ in paired_runs]
    if len(seed_values) != len(set(seed_values)):
        raise ValueError(f"Probe seeds must be unique, found {seed_values}")
    by_seed = pd.concat(by_seed_frames, ignore_index=True).sort_values(["probe_seed", "method"])

    numeric_columns = [
        "car_balanced_accuracy",
        "target_balanced_accuracy",
        "balanced_accuracy_gap",
        "target_class_recall_gap",
        "balanced_accuracy_gap_relative_reduction",
        "target_class_recall_gap_relative_reduction",
        "clean_balanced_accuracy_change",
    ]
    summary_rows = []
    for method, frame in by_seed.groupby("method", sort=False):
        row: dict[str, float | int | str] = {
            "method": method,
            "n_probe_seeds": int(frame["probe_seed"].nunique()),
        }
        for column in numeric_columns:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_std"] = float(frame[column].std(ddof=1))
        summary_rows.append(row)
    method_summary = pd.DataFrame(summary_rows)

    subjects = np.unique(paired_runs[0][1]["subjects"])
    for _, arrays in paired_runs[1:]:
        if not np.array_equal(subjects, np.unique(arrays["subjects"])):
            raise RuntimeError("All probe seeds must contain the same test subjects")
    subject_indices_by_run = [
        {subject: np.flatnonzero(arrays["subjects"] == subject) for subject in subjects}
        for _, arrays in paired_runs
    ]
    all_indices = [np.arange(arrays["y_true"].size) for _, arrays in paired_runs]
    point_by_seed = [
        _improvement_metrics(arrays, indices, target_class)
        for (_, arrays), indices in zip(paired_runs, all_indices, strict=True)
    ]
    metric_names = list(point_by_seed[0])
    point = {
        metric: float(np.mean([values[metric] for values in point_by_seed]))
        for metric in metric_names
    }
    samples = {metric: np.empty(n_resamples, dtype=np.float64) for metric in metric_names}
    rng = np.random.default_rng(bootstrap_seed)
    for bootstrap_index in range(n_resamples):
        sampled_run_indices = rng.choice(len(paired_runs), size=len(paired_runs), replace=True)
        sampled_subjects = rng.choice(subjects, size=subjects.size, replace=True)
        replicate = {metric: [] for metric in metric_names}
        for run_index in sampled_run_indices:
            arrays = paired_runs[int(run_index)][1]
            subject_indices = subject_indices_by_run[int(run_index)]
            indices = np.concatenate([subject_indices[subject] for subject in sampled_subjects])
            values = _improvement_metrics(arrays, indices, target_class)
            for metric, value in values.items():
                replicate[metric].append(value)
        for metric in metric_names:
            samples[metric][bootstrap_index] = float(np.mean(replicate[metric]))

    hierarchical = _summarize_bootstrap(
        point,
        samples,
        confidence=confidence,
        n_subjects=subjects.size,
        n_probe_seeds=len(paired_runs),
    )
    consistency_rows = by_seed[by_seed["method"] == "rule_consistency"]
    augmentation_rows = by_seed[by_seed["method"] == "multi_view_ce"]
    consistency_pass_by_seed = (
        consistency_rows["target_class_recall_gap_relative_reduction"]
        >= PRIMARY_REDUCTION_THRESHOLD
    ) & (consistency_rows["clean_balanced_accuracy_change"] >= -MAXIMUM_CLEAN_BACC_DROP)
    direction_by_seed = (
        augmentation_rows.set_index("probe_seed")["target_class_recall_gap"]
        - consistency_rows.set_index("probe_seed")["target_class_recall_gap"]
    )
    primary = hierarchical[
        hierarchical["metric"] == "target_class_recall_gap_recovery"
    ].iloc[0]
    all_consistency_passes = bool(consistency_pass_by_seed.all())
    if all_consistency_passes and float(primary["ci_lower"]) > 0.0:
        evidence_status = "supported"
    elif all_consistency_passes and float(primary["point_estimate"]) > 0.0:
        evidence_status = "promising_but_inconclusive"
    else:
        evidence_status = "not_supported"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(output_path / "method_comparison_by_seed.csv", index=False)
    method_summary.to_csv(output_path / "method_summary.csv", index=False)
    hierarchical.to_csv(output_path / "hierarchical_method_bootstrap.csv", index=False)
    aggregate_summary = {
        "probe_seeds": sorted(seed_values),
        "target_view": target_view,
        "target_class_index": int(target_class),
        "bootstrap_unit": "probe seed and paired test subject",
        "all_consistency_runs_pass_predeclared_gate": all_consistency_passes,
        "consistency_beats_augmentation_on_primary_every_seed": bool(
            (direction_by_seed > 0.0).all()
        ),
        "primary_hierarchical_bootstrap": {
            key: primary[key].item() if isinstance(primary[key], np.generic) else primary[key]
            for key in (
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "probability_improvement_positive",
                "ci_excludes_zero",
            )
        },
        "rule_loss_evidence_status": evidence_status,
        "decision_rule": (
            "supported requires every consistency run to pass the predeclared recovery/clean "
            "gate and the hierarchical 95% CI for recall-gap recovery to be above zero; a "
            "positive point estimate with a crossing CI is promising_but_inconclusive."
        ),
    }
    with (output_path / "aggregate_method_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_summary, handle, indent=2)
    return method_summary

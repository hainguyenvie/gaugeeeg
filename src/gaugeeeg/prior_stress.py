"""CPU-only known-prior calibration and small-batch stress audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .bias_manifold import (
    _fit_ridge,
    _load_predictions,
    _montage_alias,
    _reference_name,
    _validate_view_grid,
    topology_descriptor,
)
from .calibration import FittedCalibrator, expected_calibration_error, fit_calibrator


METHODS = ("identity", "topology_ridge", "prior_match", "topology_shrinkage", "oracle")


@dataclass(frozen=True)
class BatchSelection:
    condition: str
    batch_size: int
    repeat: int
    positions: NDArray[np.int64]
    construction_uses_labels: bool


def _zero_sum_matrix(n_classes: int) -> NDArray[np.float64]:
    matrix = np.zeros((n_classes, n_classes - 1), dtype=np.float64)
    matrix[:-1] = np.eye(n_classes - 1)
    matrix[-1] = -1.0
    return matrix


def _softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    return probability / probability.sum(axis=1, keepdims=True)


def fit_known_prior_bias(
    logits: NDArray[np.floating],
    prior: NDArray[np.floating] | list[float],
    *,
    l2: float = 1e-4,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> FittedCalibrator:
    """Fit bias-only calibration from logits and a known class prior.

    For additive class bias, supervised NLL depends on labels only through
    their empirical class proportions. Replacing those proportions with a
    known task prior therefore gives a label-free convex objective.
    """

    values = np.asarray(logits, dtype=np.float64)
    target_prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Known-prior calibration requires a non-empty 2-D logit array")
    if not np.isfinite(values).all():
        raise ValueError("Known-prior calibration logits must be finite")
    if target_prior.size != values.shape[1] or not np.isfinite(target_prior).all():
        raise ValueError("Prior dimension must match the logit class dimension")
    if (target_prior < 0.0).any() or not np.isclose(target_prior.sum(), 1.0):
        raise ValueError("Prior must be non-negative and sum to one")
    if l2 < 0.0 or tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("Invalid known-prior optimization settings")

    n_classes = values.shape[1]
    transform = _zero_sum_matrix(n_classes)
    free = np.zeros(n_classes - 1, dtype=np.float64)

    def objective(parameters: NDArray[np.floating]) -> float:
        bias = transform @ np.asarray(parameters, dtype=np.float64)
        shifted = values + bias
        maximum = shifted.max(axis=1, keepdims=True)
        log_partition = maximum[:, 0] + np.log(np.exp(shifted - maximum).sum(axis=1))
        return float(log_partition.mean() - target_prior @ bias + l2 * np.dot(parameters, parameters))

    current_objective = objective(free)
    success = False
    message = "maximum iterations reached"
    for iteration in range(max_iterations):
        bias = transform @ free
        probability = _softmax(values + bias)
        gradient_bias = probability.mean(axis=0) - target_prior
        gradient = transform.T @ gradient_bias + 2.0 * l2 * free
        if np.max(np.abs(gradient)) <= tolerance:
            success = True
            message = f"Newton convergence after {iteration} iterations"
            break
        hessian_bias = np.diag(probability.mean(axis=0)) - (
            probability.T @ probability / probability.shape[0]
        )
        hessian = transform.T @ hessian_bias @ transform
        hessian += 2.0 * l2 * np.eye(n_classes - 1)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        directional_decrease = float(gradient @ step)
        scale = 1.0
        accepted = False
        for _ in range(30):
            # The estimated prior may legitimately put zero mass on a class.
            # With positive L2 regularization the optimum is still finite, but
            # it can lie outside the legacy calibration bound of [-5, 5].  A
            # hard clip makes the Armijo test compare against an infeasible
            # Newton direction and reports a false line-search failure at the
            # boundary.  The objective and softmax are numerically stable, so
            # let backtracking control the unbounded Newton step instead.
            candidate = free - scale * step
            candidate_objective = objective(candidate)
            if candidate_objective <= current_objective - 1e-4 * scale * directional_decrease:
                free = candidate
                current_objective = candidate_objective
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            message = "Newton line search failed"
            break
    if not success:
        bias = transform @ free
        probability = _softmax(values + bias)
        gradient = transform.T @ (probability.mean(axis=0) - target_prior) + 2.0 * l2 * free
        success = bool(np.max(np.abs(gradient)) <= max(1e-7, 10.0 * tolerance))
        if success:
            message = "Newton convergence at final tolerance check"
    return FittedCalibrator(
        "bias",
        free,
        n_classes,
        current_objective,
        success,
        message,
    )


def _allocate_counts(prior: NDArray[np.floating], total: int) -> NDArray[np.int64]:
    raw = np.asarray(prior, dtype=np.float64) * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def _seeded_rng(seed: int, *components: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *components]))


def _stratified_positions(
    labels: NDArray[np.integer],
    prior: NDArray[np.floating],
    total: int,
    *,
    seed: int,
    components: tuple[int, ...],
) -> NDArray[np.int64]:
    counts = _allocate_counts(prior, total)
    selected: list[np.ndarray] = []
    for class_index, count in enumerate(counts):
        available = np.flatnonzero(labels == class_index)
        if count > available.size:
            raise ValueError(
                f"Stress batch needs {count} class-{class_index} trials, only {available.size} exist"
            )
        rng = _seeded_rng(seed, *components, class_index)
        selected.append(rng.choice(available, size=int(count), replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))


def _make_batch_selections(
    labels: NDArray[np.integer],
    *,
    batch_sizes: list[int],
    primary_batch_size: int,
    stress_batch_size: int,
    n_resamples: int,
    seed: int,
) -> list[BatchSelection]:
    n_trials = labels.size
    sizes = sorted(set(int(size) for size in batch_sizes))
    if not sizes or sizes[0] < 2 or sizes[-1] > n_trials:
        raise ValueError(f"batch_sizes must lie in [2, {n_trials}]")
    if sizes[-1] != n_trials:
        raise ValueError(f"batch_sizes must include the full fitting batch ({n_trials})")
    if primary_batch_size not in sizes or stress_batch_size not in sizes:
        raise ValueError("primary_batch_size and stress_batch_size must occur in batch_sizes")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    n_classes = int(np.max(labels)) + 1
    uniform = np.full(n_classes, 1.0 / n_classes)
    selections: list[BatchSelection] = []
    for size in sizes:
        repeats = 1 if size == n_trials else n_resamples
        for repeat in range(repeats):
            if size == n_trials:
                positions = np.arange(n_trials, dtype=np.int64)
            else:
                positions = np.sort(
                    _seeded_rng(seed, 1, size, repeat).choice(
                        n_trials,
                        size=size,
                        replace=False,
                    )
                ).astype(np.int64)
            selections.append(BatchSelection("random", size, repeat, positions, False))

    for size in sorted({primary_batch_size, stress_batch_size}):
        for repeat in range(n_resamples):
            positions = _stratified_positions(
                labels,
                uniform,
                size,
                seed=seed,
                components=(2, size, repeat),
            )
            selections.append(BatchSelection("balanced", size, repeat, positions, True))

    for severity_index, dominant_probability in enumerate((0.4, 0.7), start=3):
        for dominant_class in range(n_classes):
            prior = np.full(
                n_classes,
                (1.0 - dominant_probability) / (n_classes - 1),
            )
            prior[dominant_class] = dominant_probability
            condition = f"skew_{dominant_probability:.1f}_class_{dominant_class}"
            for repeat in range(n_resamples):
                positions = _stratified_positions(
                    labels,
                    prior,
                    stress_batch_size,
                    seed=seed,
                    components=(severity_index, dominant_class, stress_batch_size, repeat),
                )
                selections.append(
                    BatchSelection(condition, stress_batch_size, repeat, positions, True)
                )
    return selections


def _prediction_metrics(
    labels: NDArray[np.integer],
    logits: NDArray[np.floating],
) -> tuple[dict[str, float], NDArray[np.float64]]:
    probability = _softmax(logits)
    prediction = probability.argmax(axis=1)
    n_classes = probability.shape[1]
    recalls = np.asarray(
        [
            np.mean(prediction[labels == class_index] == class_index)
            if np.any(labels == class_index)
            else 0.0
            for class_index in range(n_classes)
        ],
        dtype=np.float64,
    )
    metrics = {
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(recalls.mean()),
        "nll": float(-np.log(np.clip(probability[np.arange(labels.size), labels], 1e-12, 1.0)).mean()),
        "ece_15": expected_calibration_error(labels, probability),
        "worst_class_recall": float(recalls.min()),
    }
    return metrics, recalls


def _apply_free_bias(
    logits: NDArray[np.floating],
    free_bias: NDArray[np.floating],
) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    transform = _zero_sum_matrix(values.shape[1])
    return values + transform @ np.asarray(free_bias, dtype=np.float64)


def _evaluate_target(
    target: pd.DataFrame,
    anchor: pd.DataFrame,
    logit_columns: list[str],
    target_bias: NDArray[np.floating] | None,
    anchor_bias: NDArray[np.floating] | None,
    oracle_bias: NDArray[np.floating],
) -> dict[str, float]:
    labels = target["y_true"].to_numpy(dtype=np.int64)
    target_logits = target[logit_columns].to_numpy(dtype=np.float64)
    anchor_logits = anchor[logit_columns].to_numpy(dtype=np.float64)
    if target_bias is not None:
        target_logits = _apply_free_bias(target_logits, target_bias)
    if anchor_bias is not None:
        anchor_logits = _apply_free_bias(anchor_logits, anchor_bias)
    metrics, target_recalls = _prediction_metrics(labels, target_logits)
    anchor_metrics, anchor_recalls = _prediction_metrics(labels, anchor_logits)
    recall_gap = np.abs(target_recalls - anchor_recalls)
    predicted = np.zeros_like(oracle_bias) if target_bias is None else np.asarray(target_bias)
    return {
        **metrics,
        "car_anchor_balanced_accuracy": anchor_metrics["balanced_accuracy"],
        "target_bias_rmse": float(np.sqrt(np.mean((predicted - oracle_bias) ** 2))),
        "max_class_recall_gap_to_car": float(recall_gap.max()),
        "mean_class_recall_gap_to_car": float(recall_gap.mean()),
    }


def _cluster_bootstrap_delta(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    metric: str,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    grouped = frame.groupby(["repeat", "held_out_reference", "method"], as_index=False)[
        metric
    ].mean()
    pivot = grouped.pivot(index=["repeat", "held_out_reference"], columns="method", values=metric)
    if candidate not in pivot or baseline not in pivot or pivot[[candidate, baseline]].isna().any().any():
        raise RuntimeError("Primary bootstrap comparison is incomplete")
    delta = (pivot[candidate] - pivot[baseline]).unstack("held_out_reference")
    values = delta.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        repeat_index = rng.integers(0, values.shape[0], size=values.shape[0])
        reference_index = rng.integers(0, values.shape[1], size=values.shape[1])
        samples[index] = values[np.ix_(repeat_index, reference_index)].mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "candidate": candidate,
        "baseline": baseline,
        "candidate_minus_baseline": float(values.mean()),
        "ci_lower": float(np.quantile(samples, alpha)),
        "ci_upper": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "n_repeats": int(values.shape[0]),
        "n_reference_clusters": int(values.shape[1]),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def analyze_prior_stress(
    validation_predictions: str | Path,
    output_dir: str | Path,
    *,
    fit_subjects: list[int],
    evaluation_subjects: list[int],
    batch_sizes: list[int],
    primary_batch_size: int = 32,
    stress_batch_size: int = 128,
    n_resamples: int = 20,
    seed: int = 20260715,
    ridge_alpha: float = 1.0,
    l2: float = 1e-4,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    minimum_rmse_reduction: float = 0.20,
    minimum_gap_reduction: float = 0.10,
    max_mean_bacc_loss: float = 0.01,
) -> pd.DataFrame:
    """Stress known-prior matching and label-free topology shrinkage."""

    if not fit_subjects or not evaluation_subjects:
        raise ValueError("fit_subjects and evaluation_subjects must be non-empty")
    if set(fit_subjects) & set(evaluation_subjects):
        raise ValueError("Fit and evaluation subjects must be disjoint")
    if ridge_alpha <= 0.0 or l2 < 0.0:
        raise ValueError("Invalid ridge or calibration regularization")
    if bootstrap_resamples < 1 or not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("Invalid bootstrap settings")

    frame, class_names = _load_predictions(validation_predictions)
    native = frame["test_view"].astype(str).str.casefold().str.startswith("native")
    frame = frame.loc[native].copy()
    if frame.empty:
        raise ValueError("No native validation views were found")
    logit_columns = [f"logit_{name}" for name in class_names]
    _validate_view_grid(frame, logit_columns)
    observed_subjects = set(frame["subject_id"].astype(int))
    requested_subjects = set(fit_subjects) | set(evaluation_subjects)
    if not requested_subjects <= observed_subjects:
        raise ValueError(f"Requested subjects are missing: {sorted(requested_subjects - observed_subjects)}")

    views = sorted(frame["test_view"].unique())
    montages = sorted({_montage_alias(view) for view in views})
    car_by_montage: dict[str, str] = {}
    for montage in montages:
        candidates = [
            view
            for view in views
            if _montage_alias(view) == montage and _reference_name(view) == "car"
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one CAR anchor for {montage}, observed {candidates}")
        car_by_montage[montage] = candidates[0]
    targets = [view for view in views if _reference_name(view) != "car"]

    fit_frame = frame.loc[frame["subject_id"].isin(fit_subjects)]
    evaluation_frame = frame.loc[frame["subject_id"].isin(evaluation_subjects)]
    fit_by_view = {
        view: fit_frame.loc[fit_frame["test_view"] == view].sort_values("trial_index").reset_index(drop=True)
        for view in views
    }
    evaluation_by_view = {
        view: evaluation_frame.loc[evaluation_frame["test_view"] == view]
        .sort_values("trial_index")
        .reset_index(drop=True)
        for view in views
    }
    canonical_fit = fit_by_view[car_by_montage[montages[0]]]
    fit_labels = canonical_fit["y_true"].to_numpy(dtype=np.int64)
    n_classes = len(class_names)
    if not np.array_equal(np.unique(fit_labels), np.arange(n_classes)):
        raise ValueError("Fit subjects must contain every class")
    empirical_counts = np.bincount(fit_labels, minlength=n_classes)
    empirical_prior = empirical_counts / empirical_counts.sum()
    known_prior = np.full(n_classes, 1.0 / n_classes)
    selections = _make_batch_selections(
        fit_labels,
        batch_sizes=batch_sizes,
        primary_batch_size=primary_batch_size,
        stress_batch_size=stress_batch_size,
        n_resamples=n_resamples,
        seed=seed,
    )

    topology = {view: topology_descriptor(view)[0] for view in views}
    oracle_bias: dict[str, np.ndarray] = {}
    known_prior_full_bias: dict[str, np.ndarray] = {}
    empirical_prior_bias: dict[str, np.ndarray] = {}
    audit_rows: list[dict[str, Any]] = []
    for view in views:
        logits = fit_by_view[view][logit_columns].to_numpy(dtype=np.float64)
        labels = fit_by_view[view]["y_true"].to_numpy(dtype=np.int64)
        oracle = fit_calibrator(logits, labels, "bias", l2=l2)
        known = fit_known_prior_bias(logits, known_prior, l2=l2)
        empirical = fit_known_prior_bias(logits, empirical_prior, l2=l2)
        if not oracle.success or not known.success or not empirical.success:
            raise RuntimeError(f"Bias optimization failed for {view}")
        oracle_bias[view] = oracle.parameters
        known_prior_full_bias[view] = known.parameters
        empirical_prior_bias[view] = empirical.parameters
        audit_rows.append(
            {
                "view": view,
                "montage": _montage_alias(view),
                "reference": _reference_name(view),
                "oracle_bias": json.dumps(oracle.parameters.tolist()),
                "known_uniform_prior_bias": json.dumps(known.parameters.tolist()),
                "empirical_prior_bias": json.dumps(empirical.parameters.tolist()),
                "known_prior_rmse_to_oracle": float(
                    np.sqrt(np.mean((known.parameters - oracle.parameters) ** 2))
                ),
                "empirical_prior_rmse_to_oracle": float(
                    np.sqrt(np.mean((empirical.parameters - oracle.parameters) ** 2))
                ),
            }
        )

    outer_topology: dict[str, np.ndarray] = {}
    for target in targets:
        held_out = _reference_name(target)
        training = [
            view for view in views if _reference_name(view) == "car" or _reference_name(view) != held_out
        ]
        outer_topology[target] = _fit_ridge(
            np.stack([topology[view] for view in training]),
            np.stack([oracle_bias[view] for view in training]),
            topology[target],
            alpha=ridge_alpha,
        )

    unique_references = sorted({_reference_name(view) for view in targets})
    topology_mse_by_outer: dict[str, float] = {}
    for outer_reference in unique_references:
        errors: list[float] = []
        source_views = [
            view for view in targets if _reference_name(view) != outer_reference
        ]
        for source in source_views:
            source_reference = _reference_name(source)
            training = [
                view
                for view in views
                if _reference_name(view) == "car"
                or _reference_name(view) not in {outer_reference, source_reference}
            ]
            predicted = _fit_ridge(
                np.stack([topology[view] for view in training]),
                np.stack([oracle_bias[view] for view in training]),
                topology[source],
                alpha=ridge_alpha,
            )
            errors.extend(np.square(predicted - oracle_bias[source]).tolist())
        topology_mse_by_outer[outer_reference] = float(np.mean(errors))

    batch_bias: dict[tuple[str, int, int, str], np.ndarray] = {}
    selection_rows: list[dict[str, Any]] = []
    for selection in selections:
        selected_labels = fit_labels[selection.positions]
        counts = np.bincount(selected_labels, minlength=n_classes)
        selection_rows.append(
            {
                "condition": selection.condition,
                "batch_size": selection.batch_size,
                "repeat": selection.repeat,
                "construction_uses_labels": selection.construction_uses_labels,
                "class_counts": json.dumps(counts.tolist()),
                "trial_indices": json.dumps(
                    canonical_fit.iloc[selection.positions]["trial_index"].astype(int).tolist()
                ),
            }
        )
        for view in views:
            logits = fit_by_view[view].iloc[selection.positions][logit_columns].to_numpy(
                dtype=np.float64
            )
            calibrator = fit_known_prior_bias(logits, known_prior, l2=l2)
            if not calibrator.success:
                raise RuntimeError(
                    f"Known-prior optimization failed for {view}, {selection.condition}, "
                    f"n={selection.batch_size}, repeat={selection.repeat}: {calibrator.message}"
                )
            batch_bias[
                (selection.condition, selection.batch_size, selection.repeat, view)
            ] = calibrator.parameters

    random_by_size = {
        size: [
            selection
            for selection in selections
            if selection.condition == "random" and selection.batch_size == size
        ]
        for size in sorted(set(int(value) for value in batch_sizes))
    }
    weight_rows: list[dict[str, Any]] = []
    weights: dict[tuple[str, int], float] = {}
    for outer_reference in unique_references:
        source_views = [
            view for view in targets if _reference_name(view) != outer_reference
        ]
        topology_mse = topology_mse_by_outer[outer_reference]
        for size, random_selections in random_by_size.items():
            errors = [
                float(
                    np.mean(
                        np.square(
                            batch_bias[("random", size, selection.repeat, view)]
                            - oracle_bias[view]
                        )
                    )
                )
                for selection in random_selections
                for view in source_views
            ]
            prior_mse = float(np.mean(errors))
            denominator = topology_mse + prior_mse
            prior_weight = 0.5 if denominator <= np.finfo(float).eps else topology_mse / denominator
            weights[(outer_reference, size)] = float(prior_weight)
            weight_rows.append(
                {
                    "held_out_reference": outer_reference,
                    "batch_size": size,
                    "prior_match_mse_from_non_target_references": prior_mse,
                    "topology_mse_from_nested_non_target_references": topology_mse,
                    "prior_match_weight": prior_weight,
                    "topology_weight": 1.0 - prior_weight,
                    "n_source_views": len(source_views),
                    "n_random_resamples": len(random_selections),
                    "target_reference_labels_used": False,
                }
            )

    static_metrics: dict[tuple[str, str], dict[str, float]] = {}
    for target in targets:
        montage = _montage_alias(target)
        anchor_view = car_by_montage[montage]
        target_eval = evaluation_by_view[target]
        anchor_eval = evaluation_by_view[anchor_view]
        static_metrics[(target, "identity")] = _evaluate_target(
            target_eval,
            anchor_eval,
            logit_columns,
            None,
            None,
            oracle_bias[target],
        )
        static_metrics[(target, "topology_ridge")] = _evaluate_target(
            target_eval,
            anchor_eval,
            logit_columns,
            outer_topology[target],
            known_prior_full_bias[anchor_view],
            oracle_bias[target],
        )
        static_metrics[(target, "oracle")] = _evaluate_target(
            target_eval,
            anchor_eval,
            logit_columns,
            oracle_bias[target],
            oracle_bias[anchor_view],
            oracle_bias[target],
        )

    metric_rows: list[dict[str, Any]] = []
    for selection in selections:
        for target in targets:
            montage = _montage_alias(target)
            held_out = _reference_name(target)
            anchor_view = car_by_montage[montage]
            target_eval = evaluation_by_view[target]
            anchor_eval = evaluation_by_view[anchor_view]
            prior_bias = batch_bias[
                (selection.condition, selection.batch_size, selection.repeat, target)
            ]
            prior_weight = weights[(held_out, selection.batch_size)]
            shrinkage_bias = (
                prior_weight * prior_bias + (1.0 - prior_weight) * outer_topology[target]
            )
            dynamic = {
                "prior_match": _evaluate_target(
                    target_eval,
                    anchor_eval,
                    logit_columns,
                    prior_bias,
                    known_prior_full_bias[anchor_view],
                    oracle_bias[target],
                ),
                "topology_shrinkage": _evaluate_target(
                    target_eval,
                    anchor_eval,
                    logit_columns,
                    shrinkage_bias,
                    known_prior_full_bias[anchor_view],
                    oracle_bias[target],
                ),
            }
            methods_for_selection = (
                METHODS
                if selection.repeat == 0
                else ("prior_match", "topology_shrinkage")
            )
            for method in methods_for_selection:
                metrics = (
                    dynamic[method]
                    if method in dynamic
                    else static_metrics[(target, method)]
                )
                metric_rows.append(
                    {
                        "condition": selection.condition,
                        "batch_size": selection.batch_size,
                        "repeat": selection.repeat,
                        "stress_construction_uses_labels": selection.construction_uses_labels,
                        "target_view": target,
                        "montage": montage,
                        "held_out_reference": held_out,
                        "method": method,
                        "target_reference_labels_used": method == "oracle",
                        "target_unlabeled_logits_used": method
                        in {"prior_match", "topology_shrinkage"},
                        "prior_match_weight": prior_weight
                        if method == "topology_shrinkage"
                        else float("nan"),
                        **metrics,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    aggregate = (
        metrics.groupby(["condition", "batch_size", "method"], as_index=False)
        .agg(
            n_rows=("target_view", "size"),
            n_target_views=("target_view", "nunique"),
            n_repeats=("repeat", "nunique"),
            mean_target_bias_rmse=("target_bias_rmse", "mean"),
            p90_target_bias_rmse=("target_bias_rmse", lambda values: values.quantile(0.90)),
            worst_target_bias_rmse=("target_bias_rmse", "max"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            worst_balanced_accuracy=("balanced_accuracy", "min"),
            mean_max_class_recall_gap=("max_class_recall_gap_to_car", "mean"),
            worst_class_recall_gap=("max_class_recall_gap_to_car", "max"),
            mean_class_recall_gap=("mean_class_recall_gap_to_car", "mean"),
            mean_nll=("nll", "mean"),
            mean_ece_15=("ece_15", "mean"),
        )
        .sort_values(["condition", "batch_size", "method"])
        .reset_index(drop=True)
    )

    primary_metrics = metrics.loc[
        (metrics["condition"] == "random")
        & (metrics["batch_size"] == primary_batch_size)
    ]
    bootstrap_rows = [
        _cluster_bootstrap_delta(
            primary_metrics,
            candidate="topology_shrinkage",
            baseline="prior_match",
            metric=metric,
            n_resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=seed + offset,
        )
        for offset, metric in enumerate(
            ["target_bias_rmse", "balanced_accuracy", "max_class_recall_gap_to_car"]
        )
    ]
    bootstrap = pd.DataFrame(bootstrap_rows)

    def aggregate_record(condition: str, size: int, method: str) -> dict[str, Any]:
        rows = aggregate.loc[
            (aggregate["condition"] == condition)
            & (aggregate["batch_size"] == size)
            & (aggregate["method"] == method)
        ]
        if rows.shape[0] != 1:
            raise RuntimeError(f"Missing aggregate row for {condition}, {size}, {method}")
        return {key: _json_value(value) for key, value in rows.iloc[0].to_dict().items()}

    primary_prior = aggregate_record("random", primary_batch_size, "prior_match")
    primary_topology = aggregate_record("random", primary_batch_size, "topology_ridge")
    primary_shrinkage = aggregate_record(
        "random", primary_batch_size, "topology_shrinkage"
    )
    shrinkage_rmse_reduction = 1.0 - (
        primary_shrinkage["mean_target_bias_rmse"]
        / max(primary_prior["mean_target_bias_rmse"], np.finfo(float).eps)
    )
    shrinkage_gap_reduction = 1.0 - (
        primary_shrinkage["mean_max_class_recall_gap"]
        / max(primary_prior["mean_max_class_recall_gap"], np.finfo(float).eps)
    )
    shrinkage_bacc_change = (
        primary_shrinkage["mean_balanced_accuracy"]
        - primary_prior["mean_balanced_accuracy"]
    )
    rmse_bootstrap = bootstrap.loc[bootstrap["metric"] == "target_bias_rmse"].iloc[0]
    shrinkage_supported = bool(
        shrinkage_rmse_reduction >= minimum_rmse_reduction
        and primary_shrinkage["mean_target_bias_rmse"]
        < primary_topology["mean_target_bias_rmse"]
        and shrinkage_gap_reduction >= minimum_gap_reduction
        and shrinkage_bacc_change >= -max_mean_bacc_loss
        and rmse_bootstrap["ci_upper"] < 0.0
    )

    full_batch_size = max(int(value) for value in batch_sizes)
    full_prior = aggregate_record("random", full_batch_size, "prior_match")
    full_oracle = aggregate_record("random", full_batch_size, "oracle")
    full_bias_audit = pd.DataFrame(audit_rows)
    empirical_equivalence_rmse = float(
        full_bias_audit["empirical_prior_rmse_to_oracle"].mean()
    )
    known_prior_rmse = float(full_bias_audit["known_prior_rmse_to_oracle"].mean())
    known_prior_reproduces_oracle = bool(
        known_prior_rmse <= 0.01
        and full_prior["mean_balanced_accuracy"]
        >= full_oracle["mean_balanced_accuracy"] - 0.005
    )

    balanced_stress = aggregate_record("balanced", stress_batch_size, "prior_match")
    severe_rows = aggregate.loc[
        aggregate["condition"].str.startswith("skew_0.7_")
        & (aggregate["batch_size"] == stress_batch_size)
        & (aggregate["method"] == "prior_match")
    ]
    if severe_rows.shape[0] != n_classes:
        raise RuntimeError("Severe-prior stress suite is incomplete")
    severe_rmse = float(severe_rows["mean_target_bias_rmse"].mean())
    severe_bacc = float(severe_rows["mean_balanced_accuracy"].mean())
    severe_gap = float(severe_rows["mean_max_class_recall_gap"].mean())
    prior_confounding = {
        "balanced_batch_size": stress_batch_size,
        "balanced_mean_target_bias_rmse": balanced_stress["mean_target_bias_rmse"],
        "severe_skew_mean_target_bias_rmse": severe_rmse,
        "severe_to_balanced_rmse_ratio": severe_rmse
        / max(balanced_stress["mean_target_bias_rmse"], np.finfo(float).eps),
        "severe_minus_balanced_mean_bacc": severe_bacc
        - balanced_stress["mean_balanced_accuracy"],
        "severe_minus_balanced_mean_max_recall_gap": severe_gap
        - balanced_stress["mean_max_class_recall_gap"],
    }
    prior_confounding_detected = bool(
        prior_confounding["severe_to_balanced_rmse_ratio"] >= 2.0
        or prior_confounding["severe_minus_balanced_mean_bacc"] <= -0.03
        or prior_confounding["severe_minus_balanced_mean_max_recall_gap"] >= 0.05
    )

    if shrinkage_supported:
        recommendation = "develop_operator_regularized_small_batch_prior_matching"
    elif known_prior_reproduces_oracle and prior_confounding_detected:
        recommendation = "use_prior_matching_only_with_known_or_controlled_class_prior"
    elif known_prior_reproduces_oracle:
        recommendation = "use_known_prior_matching_without_learned_manifold"
    else:
        recommendation = "known_prior_matching_not_sufficient"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "prior_stress_metrics.csv", index=False)
    aggregate.to_csv(output / "prior_stress_aggregate.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "prior_stress_weights.csv", index=False)
    full_bias_audit.to_csv(output / "prior_stress_bias_audit.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "prior_stress_selections.csv", index=False)
    bootstrap.to_csv(output / "prior_stress_bootstrap.csv", index=False)
    summary = {
        "stage": "E10 known-prior and small-batch stress audit",
        "fit_subjects": sorted(set(int(value) for value in fit_subjects)),
        "evaluation_subjects": sorted(set(int(value) for value in evaluation_subjects)),
        "fit_evaluation_subjects_disjoint": not bool(
            set(fit_subjects) & set(evaluation_subjects)
        ),
        "physionet_test_subjects_used": False,
        "n_views": len(views),
        "n_target_views": len(targets),
        "held_out_unit": "reference electrode identity across all montages",
        "class_names": class_names,
        "fit_class_counts": empirical_counts.tolist(),
        "empirical_fit_prior": empirical_prior.tolist(),
        "deployable_known_prior": known_prior.tolist(),
        "prior_matching_identity": (
            "For additive bias, supervised NLL depends on labels only through class "
            "proportions; replacing them with a known task prior is label-free."
        ),
        "empirical_prior_match_mean_rmse_to_supervised_oracle": empirical_equivalence_rmse,
        "known_uniform_prior_match_mean_rmse_to_supervised_oracle": known_prior_rmse,
        "known_prior_matching_reproduces_oracle": known_prior_reproduces_oracle,
        "batch_sizes": sorted(set(int(value) for value in batch_sizes)),
        "primary_random_batch_size": primary_batch_size,
        "prior_stress_batch_size": stress_batch_size,
        "batch_resamples": n_resamples,
        "stress_batch_labels_used_for_construction_only": True,
        "held_out_target_labels_used_for_candidate_fitting": False,
        "shrinkage_weight_tuned_on_non_target_references_only": True,
        "ridge_alpha_predeclared": ridge_alpha,
        "l2_predeclared": l2,
        "primary_prior_match": primary_prior,
        "primary_topology_ridge": primary_topology,
        "primary_topology_shrinkage": primary_shrinkage,
        "primary_shrinkage_rmse_reduction_vs_prior_match": shrinkage_rmse_reduction,
        "primary_shrinkage_gap_reduction_vs_prior_match": shrinkage_gap_reduction,
        "primary_shrinkage_mean_bacc_change_vs_prior_match": shrinkage_bacc_change,
        "primary_bootstrap": [
            {key: _json_value(value) for key, value in row.items()}
            for row in bootstrap.to_dict(orient="records")
        ],
        "topology_shrinkage_supported": shrinkage_supported,
        "full_batch_prior_match": full_prior,
        "full_batch_oracle": full_oracle,
        "prior_confounding": prior_confounding,
        "prior_confounding_detected": prior_confounding_detected,
        "selection_rule": (
            f"At random n={primary_batch_size}, topology shrinkage must reduce mean bias "
            f"RMSE by >= {minimum_rmse_reduction:.0%} versus prior matching, beat topology "
            f"RMSE, reduce mean maximum recall gap by >= {minimum_gap_reduction:.0%}, lose "
            f"no more than {max_mean_bacc_loss:.3f} mean BAcc, and have a paired "
            f"{bootstrap_confidence:.0%} RMSE-delta interval below zero."
        ),
        "next_method_recommendation": recommendation,
    }
    with (output / "prior_stress_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return aggregate

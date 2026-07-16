"""Source-only class/operator trust safeguard for prior-shift correction."""

from __future__ import annotations

import json
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
from .calibration import fit_calibrator
from .prior_identifiability import (
    _aggregate_metrics,
    _center_logits,
    _fit_soft_prior_model,
    _mean_record,
    analyze_prior_identifiability,
    estimate_regularized_soft_prior,
)
from .prior_stress import (
    _apply_free_bias,
    _cluster_bootstrap_delta,
    _evaluate_target,
    _json_value,
    _make_batch_selections,
    _zero_sum_matrix,
    fit_known_prior_bias,
)


SAFE_METHOD = "class_operator_trust_safeguard"
BASELINE_METHOD = "fixed_topology_shrinkage"
E11_METHOD = "operator_confusion_shrinkage"
TOPOLOGY_METHOD = "topology_ridge"


def _parse_vector(value: str | NDArray[np.floating]) -> NDArray[np.float64]:
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def fit_class_trust_caps(
    topology_bias: NDArray[np.floating],
    raw_candidate_bias: NDArray[np.floating],
    oracle_bias: NDArray[np.floating],
    *,
    n_classes: int,
    ridge: float = 1e-8,
) -> NDArray[np.float64]:
    """Fit diagonal class trust by source-only least squares in full bias space."""

    topology = np.asarray(topology_bias, dtype=np.float64)
    candidate = np.asarray(raw_candidate_bias, dtype=np.float64)
    oracle = np.asarray(oracle_bias, dtype=np.float64)
    expected = (topology.shape[0], n_classes - 1)
    if topology.ndim != 2 or topology.shape != expected:
        raise ValueError("topology_bias has an invalid free-bias shape")
    if candidate.shape != topology.shape or oracle.shape != topology.shape:
        raise ValueError("Trust-cap bias arrays must have matching shapes")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    transform = _zero_sum_matrix(n_classes)
    topology_full = topology @ transform.T
    delta = candidate @ transform.T - topology_full
    target = oracle @ transform.T - topology_full
    denominator = np.square(delta).sum(axis=0) + ridge
    caps = (delta * target).sum(axis=0) / denominator
    return np.clip(caps, 0.0, 1.0)


def apply_class_trust_caps(
    topology_bias: NDArray[np.floating],
    candidate_bias: NDArray[np.floating],
    *,
    base_weight: float,
    class_caps: NDArray[np.floating],
    n_classes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cap E11's scalar trust component-wise and return a free zero-sum bias."""

    topology = np.asarray(topology_bias, dtype=np.float64)
    candidate = np.asarray(candidate_bias, dtype=np.float64)
    caps = np.asarray(class_caps, dtype=np.float64).reshape(-1)
    if topology.shape != (n_classes - 1,) or candidate.shape != topology.shape:
        raise ValueError("Expected one free-bias vector with n_classes - 1 entries")
    if caps.shape != (n_classes,) or (caps < 0.0).any() or (caps > 1.0).any():
        raise ValueError("class_caps must contain one value in [0, 1] per class")
    if not 0.0 <= base_weight <= 1.0:
        raise ValueError("base_weight must be in [0, 1]")

    transform = _zero_sum_matrix(n_classes)
    inverse = np.linalg.pinv(transform)
    topology_full = transform @ topology
    if base_weight <= np.finfo(float).eps:
        raw_candidate = topology.copy()
    else:
        raw_candidate = topology + (candidate - topology) / base_weight
    candidate_full = transform @ raw_candidate
    applied_weight = np.minimum(base_weight, caps)
    safe_full = topology_full + applied_weight * (candidate_full - topology_full)
    safe_full -= safe_full.mean()
    return inverse @ safe_full, applied_weight


def _source_gate_selections(
    labels: NDArray[np.integer],
    *,
    primary_batch_size: int,
    stress_batch_size: int,
    n_resamples: int,
    seed: int,
) -> list[Any]:
    selections = _make_batch_selections(
        labels,
        batch_sizes=sorted(
            {primary_batch_size, stress_batch_size, int(labels.size)}
        ),
        primary_batch_size=primary_batch_size,
        stress_batch_size=stress_batch_size,
        n_resamples=n_resamples,
        seed=seed,
    )
    return [
        selection
        for selection in selections
        if (
            selection.condition == "random"
            and selection.batch_size == primary_batch_size
        )
        or (
            selection.condition == "balanced"
            and selection.batch_size == stress_batch_size
        )
        or (
            selection.condition.startswith("skew_0.7_")
            and selection.batch_size == stress_batch_size
        )
    ]


def _fit_source_only_safeguards(
    frame: pd.DataFrame,
    class_names: list[str],
    *,
    source_subjects: list[int],
    primary_batch_size: int,
    stress_batch_size: int,
    n_resamples: int,
    source_seed: int,
    gate_seed: int,
    ridge_alpha: float,
    l2: float,
    confusion_regularization: float,
    cap_ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logit_columns = [f"logit_{name}" for name in class_names]
    views = sorted(frame["test_view"].unique())
    montages = sorted({_montage_alias(view) for view in views})
    car_by_montage = {
        montage: next(
            view
            for view in views
            if _montage_alias(view) == montage and _reference_name(view) == "car"
        )
        for montage in montages
    }
    source = frame.loc[frame["subject_id"].isin(source_subjects)]
    source_by_view = {
        view: source.loc[source["test_view"] == view]
        .sort_values("trial_index")
        .reset_index(drop=True)
        for view in views
    }
    labels = source_by_view[car_by_montage[montages[0]]]["y_true"].to_numpy(
        dtype=np.int64
    )
    n_classes = len(class_names)
    nominal_prior = np.full(n_classes, 1.0 / n_classes)
    selections = _source_gate_selections(
        labels,
        primary_batch_size=primary_batch_size,
        stress_batch_size=stress_batch_size,
        n_resamples=n_resamples,
        seed=gate_seed,
    )

    descriptor = {view: topology_descriptor(view)[0] for view in views}
    oracle_bias: dict[str, NDArray[np.float64]] = {}
    for view in views:
        selected = source_by_view[view]
        fitted = fit_calibrator(
            selected[logit_columns].to_numpy(dtype=np.float64),
            selected["y_true"].to_numpy(dtype=np.int64),
            "bias",
            l2=l2,
        )
        if not fitted.success:
            raise RuntimeError(f"Source oracle bias failed for {view}")
        oracle_bias[view] = fitted.parameters

    prior_models = {
        montage: _fit_soft_prior_model(
            source_by_view[car_by_montage[montage]],
            logit_columns,
            source_subjects,
            seed=source_seed,
        )
        for montage in montages
    }
    targets = [view for view in views if _reference_name(view) != "car"]
    references = sorted({_reference_name(view) for view in targets})
    example_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []

    for outer_reference in references:
        outer_rows: list[dict[str, Any]] = []
        for source_view in targets:
            source_reference = _reference_name(source_view)
            if source_reference == outer_reference:
                continue
            montage = _montage_alias(source_view)
            training_views = [
                view
                for view in views
                if _reference_name(view) == "car"
                or _reference_name(view)
                not in {outer_reference, source_reference}
            ]
            topology_bias = _fit_ridge(
                np.stack([descriptor[view] for view in training_views]),
                np.stack([oracle_bias[view] for view in training_views]),
                descriptor[source_view],
                alpha=ridge_alpha,
            )
            for selection in selections:
                batch_logits = source_by_view[source_view].iloc[
                    selection.positions
                ][logit_columns].to_numpy(dtype=np.float64)
                corrected = _apply_free_bias(batch_logits, topology_bias)
                probability = prior_models[montage].estimator.predict_proba(
                    _center_logits(corrected)
                )
                observed = probability.mean(axis=0)
                prior = estimate_regularized_soft_prior(
                    prior_models[montage].soft_confusion,
                    observed,
                    nominal_prior,
                    regularization=confusion_regularization,
                )
                raw_candidate = fit_known_prior_bias(batch_logits, prior, l2=l2)
                if not raw_candidate.success:
                    raise RuntimeError(
                        "Source safeguard candidate failed for "
                        f"{outer_reference}, {source_view}, {selection.condition}"
                    )
                row = {
                    "held_out_reference": outer_reference,
                    "source_view": source_view,
                    "montage": montage,
                    "condition": selection.condition,
                    "batch_size": selection.batch_size,
                    "repeat": selection.repeat,
                    "topology_bias": json.dumps(topology_bias.tolist()),
                    "raw_candidate_bias": json.dumps(
                        raw_candidate.parameters.tolist()
                    ),
                    "source_oracle_bias": json.dumps(
                        oracle_bias[source_view].tolist()
                    ),
                    "target_reference_labels_used": False,
                }
                outer_rows.append(row)
                example_rows.append(row)

        topology = np.stack(
            [_parse_vector(row["topology_bias"]) for row in outer_rows]
        )
        raw_candidate = np.stack(
            [_parse_vector(row["raw_candidate_bias"]) for row in outer_rows]
        )
        source_oracle = np.stack(
            [_parse_vector(row["source_oracle_bias"]) for row in outer_rows]
        )
        caps = fit_class_trust_caps(
            topology,
            raw_candidate,
            source_oracle,
            n_classes=n_classes,
            ridge=cap_ridge,
        )
        transform = _zero_sum_matrix(n_classes)
        inverse = np.linalg.pinv(transform)
        topology_full = topology @ transform.T
        candidate_full = raw_candidate @ transform.T
        safe_full = topology_full + caps * (candidate_full - topology_full)
        safe_full -= safe_full.mean(axis=1, keepdims=True)
        safe_free = safe_full @ inverse.T
        topology_rmse = np.sqrt(np.mean(np.square(topology - source_oracle), axis=1))
        safe_rmse = np.sqrt(np.mean(np.square(safe_free - source_oracle), axis=1))
        conditions = np.asarray([row["condition"] for row in outer_rows])
        condition_audit = {
            condition: {
                "topology_mean_rmse": float(
                    topology_rmse[conditions == condition].mean()
                ),
                "safe_mean_rmse": float(safe_rmse[conditions == condition].mean()),
            }
            for condition in sorted(set(conditions))
        }
        cap_rows.append(
            {
                "held_out_reference": outer_reference,
                "n_source_examples": len(outer_rows),
                "n_source_views": len(
                    {row["source_view"] for row in outer_rows}
                ),
                "source_gate_resamples": n_resamples,
                "class_trust_caps": json.dumps(caps.tolist()),
                **{
                    f"cap_{class_name}": caps[class_index]
                    for class_index, class_name in enumerate(class_names)
                },
                "source_condition_audit": json.dumps(condition_audit),
                "all_source_conditions_improve_topology": all(
                    values["safe_mean_rmse"]
                    < values["topology_mean_rmse"]
                    for values in condition_audit.values()
                ),
                "target_reference_labels_used": False,
            }
        )
    return pd.DataFrame(example_rows), pd.DataFrame(cap_rows)


def _selected_metrics(
    metrics: pd.DataFrame,
    *,
    condition: str | None = None,
    condition_prefix: str | None = None,
    batch_size: int,
) -> pd.DataFrame:
    selected = metrics.loc[metrics["batch_size"] == batch_size]
    if condition is not None:
        selected = selected.loc[selected["condition"] == condition]
    if condition_prefix is not None:
        selected = selected.loc[
            selected["condition"].str.startswith(condition_prefix)
        ]
    return selected


def analyze_class_safeguard(
    validation_predictions: str | Path,
    output_dir: str | Path,
    *,
    source_subjects: list[int],
    adaptation_subjects: list[int],
    evaluation_subjects: list[int],
    batch_sizes: list[int],
    primary_batch_size: int = 32,
    stress_batch_size: int = 128,
    batch_resamples: int = 20,
    source_gate_resamples: int = 5,
    source_seed: int = 20260716,
    adaptation_seed: int = 20260717,
    gate_seed: int = 20260718,
    ridge_alpha: float = 1.0,
    l2: float = 1e-4,
    confusion_regularization: float = 1.0,
    cap_ridge: float = 1e-8,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    minimum_severe_rmse_reduction: float = 0.05,
    max_primary_rmse_increase: float = 0.05,
    max_mean_bacc_loss: float = 0.01,
    max_mean_gap_increase: float = 0.01,
) -> pd.DataFrame:
    """Fit a source-only class safeguard and audit it on disjoint subjects."""

    source_set = set(source_subjects)
    adaptation_set = set(adaptation_subjects)
    evaluation_set = set(evaluation_subjects)
    if not source_set or not adaptation_set or not evaluation_set:
        raise ValueError("All E12 subject groups must be non-empty")
    if source_set & adaptation_set or source_set & evaluation_set:
        raise ValueError("Source subjects must be disjoint from adaptation/evaluation")
    if adaptation_set & evaluation_set:
        raise ValueError("Adaptation and evaluation subjects must be disjoint")
    if source_gate_resamples < 1 or cap_ridge < 0.0:
        raise ValueError("Invalid source-gate settings")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline_dir = output / "strict_prior_baseline"
    analyze_prior_identifiability(
        validation_predictions,
        baseline_dir,
        topology_subjects=source_subjects,
        prior_model_subjects=source_subjects,
        adaptation_subjects=adaptation_subjects,
        evaluation_subjects=evaluation_subjects,
        batch_sizes=batch_sizes,
        primary_batch_size=primary_batch_size,
        stress_batch_size=stress_batch_size,
        n_resamples=batch_resamples,
        source_seed=source_seed,
        adaptation_seed=adaptation_seed,
        ridge_alpha=ridge_alpha,
        l2=l2,
        confusion_regularization=confusion_regularization,
        weak_confusion_regularization=0.1,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
        max_primary_rmse_increase=max_primary_rmse_increase,
        minimum_severe_rmse_reduction=minimum_severe_rmse_reduction,
        max_mean_bacc_loss=max_mean_bacc_loss,
        max_mean_gap_increase=max_mean_gap_increase,
    )

    frame, class_names = _load_predictions(validation_predictions)
    frame = frame.loc[
        frame["test_view"].astype(str).str.casefold().str.startswith("native")
    ].copy()
    logit_columns = [f"logit_{name}" for name in class_names]
    _validate_view_grid(frame, logit_columns)
    observed_subjects = set(frame["subject_id"].astype(int))
    requested = source_set | adaptation_set | evaluation_set
    if not requested <= observed_subjects:
        raise ValueError(
            f"Requested subjects are missing: {sorted(requested - observed_subjects)}"
        )

    source_examples, caps = _fit_source_only_safeguards(
        frame,
        class_names,
        source_subjects=source_subjects,
        primary_batch_size=primary_batch_size,
        stress_batch_size=stress_batch_size,
        n_resamples=source_gate_resamples,
        source_seed=source_seed,
        gate_seed=gate_seed,
        ridge_alpha=ridge_alpha,
        l2=l2,
        confusion_regularization=confusion_regularization,
        cap_ridge=cap_ridge,
    )
    cap_by_reference = {
        row.held_out_reference: np.asarray(
            json.loads(row.class_trust_caps), dtype=np.float64
        )
        for row in caps.itertuples(index=False)
    }

    estimates = pd.read_csv(
        baseline_dir / "prior_identifiability_estimates.csv"
    )
    required_vectors = {
        "topology_bias",
        "candidate_bias",
        "oracle_bias_audit_only",
    }
    if not required_vectors <= set(estimates.columns):
        raise RuntimeError("Strict E11 baseline did not expose safeguard vectors")

    views = sorted(frame["test_view"].unique())
    montages = sorted({_montage_alias(view) for view in views})
    car_by_montage = {
        montage: next(
            view
            for view in views
            if _montage_alias(view) == montage and _reference_name(view) == "car"
        )
        for montage in montages
    }
    source = frame.loc[frame["subject_id"].isin(source_subjects)]
    evaluation = frame.loc[frame["subject_id"].isin(evaluation_subjects)]
    source_by_view = {
        view: source.loc[source["test_view"] == view]
        .sort_values("trial_index")
        .reset_index(drop=True)
        for view in views
    }
    evaluation_by_view = {
        view: evaluation.loc[evaluation["test_view"] == view]
        .sort_values("trial_index")
        .reset_index(drop=True)
        for view in views
    }
    nominal_prior = np.full(len(class_names), 1.0 / len(class_names))
    anchor_bias: dict[str, NDArray[np.float64]] = {}
    for montage, car_view in car_by_montage.items():
        fitted = fit_known_prior_bias(
            source_by_view[car_view][logit_columns].to_numpy(dtype=np.float64),
            nominal_prior,
            l2=l2,
        )
        if not fitted.success:
            raise RuntimeError(f"Source CAR anchor failed for {montage}")
        anchor_bias[montage] = fitted.parameters

    safe_metric_rows: list[dict[str, Any]] = []
    trust_rows: list[dict[str, Any]] = []
    for row in estimates.itertuples(index=False):
        topology_bias = _parse_vector(row.topology_bias)
        candidate_bias = _parse_vector(row.candidate_bias)
        oracle_bias = _parse_vector(row.oracle_bias_audit_only)
        class_caps = cap_by_reference[row.held_out_reference]
        safe_bias, applied_weight = apply_class_trust_caps(
            topology_bias,
            candidate_bias,
            base_weight=float(row.prior_match_weight),
            class_caps=class_caps,
            n_classes=len(class_names),
        )
        target = evaluation_by_view[row.target_view]
        anchor_view = car_by_montage[row.montage]
        metrics = _evaluate_target(
            target,
            evaluation_by_view[anchor_view],
            logit_columns,
            safe_bias,
            anchor_bias[row.montage],
            oracle_bias,
        )
        safe_metric_rows.append(
            {
                "condition": row.condition,
                "batch_size": row.batch_size,
                "repeat": row.repeat,
                "stress_construction_uses_labels": (
                    row.stress_construction_uses_labels
                ),
                "target_view": row.target_view,
                "montage": row.montage,
                "held_out_reference": row.held_out_reference,
                "method": SAFE_METHOD,
                "target_reference_labels_used": False,
                "target_unlabeled_logits_used": True,
                "prior_match_weight": row.prior_match_weight,
                **metrics,
            }
        )
        trust_rows.append(
            {
                "condition": row.condition,
                "batch_size": row.batch_size,
                "repeat": row.repeat,
                "target_view": row.target_view,
                "montage": row.montage,
                "held_out_reference": row.held_out_reference,
                "base_prior_weight": row.prior_match_weight,
                "class_trust_caps": json.dumps(class_caps.tolist()),
                "applied_class_weights": json.dumps(applied_weight.tolist()),
                "safe_bias": json.dumps(safe_bias.tolist()),
                "target_reference_labels_used": False,
            }
        )

    baseline_metrics = pd.read_csv(
        baseline_dir / "prior_identifiability_metrics.csv"
    )
    metrics = pd.concat(
        [baseline_metrics, pd.DataFrame(safe_metric_rows)], ignore_index=True
    )
    aggregate = _aggregate_metrics(metrics)

    comparison_specs = [
        (
            "primary_random",
            _selected_metrics(
                metrics,
                condition="random",
                batch_size=primary_batch_size,
            ),
        ),
        (
            "balanced_stress_size",
            _selected_metrics(
                metrics,
                condition="balanced",
                batch_size=stress_batch_size,
            ),
        ),
        (
            "severe_skew",
            _selected_metrics(
                metrics,
                condition_prefix="skew_0.7_",
                batch_size=stress_batch_size,
            ),
        ),
    ]
    bootstrap_rows: list[dict[str, Any]] = []
    for comparison_index, (comparison, selected) in enumerate(comparison_specs):
        for metric_index, metric in enumerate(
            [
                "target_bias_rmse",
                "balanced_accuracy",
                "max_class_recall_gap_to_car",
            ]
        ):
            result = _cluster_bootstrap_delta(
                selected,
                candidate=SAFE_METHOD,
                baseline=BASELINE_METHOD,
                metric=metric,
                n_resamples=bootstrap_resamples,
                confidence=bootstrap_confidence,
                seed=gate_seed + 10 * comparison_index + metric_index,
            )
            result["comparison"] = comparison
            bootstrap_rows.append(result)

    for class_index, class_name in enumerate(class_names):
        condition = f"skew_0.7_class_{class_index}"
        selected = _selected_metrics(
            metrics,
            condition=condition,
            batch_size=stress_batch_size,
        )
        result = _cluster_bootstrap_delta(
            selected,
            candidate=SAFE_METHOD,
            baseline=BASELINE_METHOD,
            metric="target_bias_rmse",
            n_resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=gate_seed + 100 + class_index,
        )
        result["comparison"] = f"severe_{class_name}"
        bootstrap_rows.append(result)
    bootstrap = pd.DataFrame(bootstrap_rows)

    primary_fixed = _mean_record(
        metrics,
        condition="random",
        batch_size=primary_batch_size,
        method=BASELINE_METHOD,
    )
    primary_safe = _mean_record(
        metrics,
        condition="random",
        batch_size=primary_batch_size,
        method=SAFE_METHOD,
    )
    balanced_fixed = _mean_record(
        metrics,
        condition="balanced",
        batch_size=stress_batch_size,
        method=BASELINE_METHOD,
    )
    balanced_safe = _mean_record(
        metrics,
        condition="balanced",
        batch_size=stress_batch_size,
        method=SAFE_METHOD,
    )
    severe_fixed = _mean_record(
        metrics,
        condition_prefix="skew_0.7_",
        batch_size=stress_batch_size,
        method=BASELINE_METHOD,
    )
    severe_safe = _mean_record(
        metrics,
        condition_prefix="skew_0.7_",
        batch_size=stress_batch_size,
        method=SAFE_METHOD,
    )
    severe_topology = _mean_record(
        metrics,
        condition_prefix="skew_0.7_",
        batch_size=stress_batch_size,
        method=TOPOLOGY_METHOD,
    )
    severe_reduction = 1.0 - severe_safe["mean_target_bias_rmse"] / max(
        severe_fixed["mean_target_bias_rmse"], np.finfo(float).eps
    )
    primary_preserved = bool(
        primary_safe["mean_target_bias_rmse"]
        <= primary_fixed["mean_target_bias_rmse"]
        * (1.0 + max_primary_rmse_increase)
        and primary_safe["mean_balanced_accuracy"]
        >= primary_fixed["mean_balanced_accuracy"] - max_mean_bacc_loss
        and primary_safe["mean_max_class_recall_gap"]
        <= primary_fixed["mean_max_class_recall_gap"] + max_mean_gap_increase
    )
    balanced_preserved = bool(
        balanced_safe["mean_target_bias_rmse"]
        <= balanced_fixed["mean_target_bias_rmse"]
        * (1.0 + max_primary_rmse_increase)
        and balanced_safe["mean_balanced_accuracy"]
        >= balanced_fixed["mean_balanced_accuracy"] - max_mean_bacc_loss
        and balanced_safe["mean_max_class_recall_gap"]
        <= balanced_fixed["mean_max_class_recall_gap"] + max_mean_gap_increase
    )
    severe_bootstrap = bootstrap.loc[
        (bootstrap["comparison"] == "severe_skew")
        & (bootstrap["metric"] == "target_bias_rmse")
    ].iloc[0]
    mean_severe_supported = bool(
        severe_reduction >= minimum_severe_rmse_reduction
        and severe_safe["mean_target_bias_rmse"]
        < severe_topology["mean_target_bias_rmse"]
        and severe_safe["mean_balanced_accuracy"]
        >= severe_fixed["mean_balanced_accuracy"] - max_mean_bacc_loss
        and severe_safe["mean_max_class_recall_gap"]
        <= severe_fixed["mean_max_class_recall_gap"] + max_mean_gap_increase
        and severe_bootstrap["ci_upper"] < 0.0
    )

    per_class: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        condition = f"skew_0.7_class_{class_index}"
        fixed = _mean_record(
            metrics,
            condition=condition,
            batch_size=stress_batch_size,
            method=BASELINE_METHOD,
        )
        safe = _mean_record(
            metrics,
            condition=condition,
            batch_size=stress_batch_size,
            method=SAFE_METHOD,
        )
        interval = bootstrap.loc[
            bootstrap["comparison"] == f"severe_{class_name}"
        ].iloc[0]
        per_class.append(
            {
                "dominant_class": class_index,
                "class_name": class_name,
                "fixed_mean_bias_rmse": fixed["mean_target_bias_rmse"],
                "safe_mean_bias_rmse": safe["mean_target_bias_rmse"],
                "safe_minus_fixed": (
                    safe["mean_target_bias_rmse"]
                    - fixed["mean_target_bias_rmse"]
                ),
                "cluster_delta": float(interval["candidate_minus_baseline"]),
                "cluster_ci_lower": float(interval["ci_lower"]),
                "cluster_ci_upper": float(interval["ci_upper"]),
            }
        )
    all_class_point_improve = all(row["safe_minus_fixed"] < 0.0 for row in per_class)
    all_class_cluster_point_improve = all(
        row["cluster_delta"] < 0.0 for row in per_class
    )
    class_harm_not_detected = all(
        row["cluster_ci_lower"] <= 0.0 for row in per_class
    )
    class_improvement_confirmed = all(
        row["cluster_ci_upper"] < 0.0 for row in per_class
    )
    confirmation_supported = bool(
        primary_preserved
        and balanced_preserved
        and mean_severe_supported
        and all_class_point_improve
        and all_class_cluster_point_improve
        and class_harm_not_detected
    )
    paper_claim_supported = bool(
        confirmation_supported and class_improvement_confirmed
    )

    baseline_summary = json.loads(
        (baseline_dir / "prior_identifiability_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "stage": "E12 source-only class/operator trust safeguard",
        "source_subjects": sorted(source_set),
        "adaptation_subjects": sorted(adaptation_set),
        "evaluation_subjects": sorted(evaluation_set),
        "source_adaptation_subjects_disjoint": not bool(
            source_set & adaptation_set
        ),
        "source_evaluation_subjects_disjoint": not bool(
            source_set & evaluation_set
        ),
        "adaptation_evaluation_subjects_disjoint": not bool(
            adaptation_set & evaluation_set
        ),
        "physionet_test_subjects_used": False,
        "held_out_unit": "reference electrode identity across all montages",
        "target_reference_labels_used_for_caps": False,
        "target_reference_labels_used_for_safe_bias": False,
        "source_labels_used_for_controlled_gate_training": True,
        "adaptation_labels_used_for_stress_construction_and_audit_only": True,
        "class_names": class_names,
        "batch_sizes": sorted(set(int(value) for value in batch_sizes)),
        "primary_batch_size": primary_batch_size,
        "stress_batch_size": stress_batch_size,
        "batch_resamples": batch_resamples,
        "source_gate_resamples": source_gate_resamples,
        "source_seed": source_seed,
        "adaptation_seed": adaptation_seed,
        "gate_seed": gate_seed,
        "cap_ridge": cap_ridge,
        "cap_selection": (
            "Per held-out reference, diagonal least-squares trust in full "
            "four-class zero-sum bias space; target reference excluded from "
            "both source examples and nested topology fits; caps clipped to [0,1]."
        ),
        "all_source_folds_improve_every_training_condition": bool(
            caps["all_source_conditions_improve_topology"].all()
        ),
        "mean_class_caps": {
            class_name: float(caps[f"cap_{class_name}"].mean())
            for class_name in class_names
        },
        "strict_e11_baseline": {
            "topology_adaptation_subjects_disjoint": baseline_summary[
                "topology_adaptation_subjects_disjoint"
            ],
            "mean_severe_robustness_supported": baseline_summary[
                "mean_severe_robustness_supported"
            ],
            "operator_confusion_shrinkage_supported": baseline_summary[
                "operator_confusion_shrinkage_supported"
            ],
        },
        "primary_fixed": primary_fixed,
        "primary_safe": primary_safe,
        "primary_preserved": primary_preserved,
        "balanced_fixed": balanced_fixed,
        "balanced_safe": balanced_safe,
        "balanced_preserved": balanced_preserved,
        "severe_fixed": severe_fixed,
        "severe_safe": severe_safe,
        "severe_topology": severe_topology,
        "severe_rmse_reduction_vs_fixed": severe_reduction,
        "mean_severe_robustness_supported": mean_severe_supported,
        "severe_per_dominant_class": per_class,
        "all_dominant_class_point_directions_improve": bool(
            all_class_point_improve
        ),
        "all_dominant_class_cluster_point_directions_improve": bool(
            all_class_cluster_point_improve
        ),
        "class_harm_detected_by_two_sided_interval": not class_harm_not_detected,
        "all_class_improvements_individually_confirmed": bool(
            class_improvement_confirmed
        ),
        "safeguard_supported_for_repeated_seed_confirmation": (
            confirmation_supported
        ),
        "paper_level_class_uniform_claim_supported": paper_claim_supported,
        "bootstrap": [
            {key: _json_value(value) for key, value in row.items()}
            for row in bootstrap.to_dict(orient="records")
        ],
        "selection_rule": (
            "Preserve random and balanced conditions; reduce severe mean RMSE "
            f"by >= {minimum_severe_rmse_reduction:.0%}, beat topology-only, "
            "obtain a pooled paired interval below zero, improve every raw and "
            "reference-clustered class direction, and detect no class-specific "
            "harm. A paper-level class-uniform claim additionally requires every "
            "class-specific interval to lie below zero."
        ),
        "next_method_recommendation": (
            "repeat_seeds_and_external_dataset_before_class_uniform_claim"
            if confirmation_supported and not paper_claim_supported
            else "advance_class_operator_safeguard_to_confirmation"
            if paper_claim_supported
            else "add_beyond_logits_signal_or_stronger_source_safety_constraint"
        ),
    }

    source_examples.to_csv(
        output / "class_safeguard_source_examples.csv", index=False
    )
    caps.to_csv(output / "class_safeguard_caps.csv", index=False)
    pd.DataFrame(trust_rows).to_csv(
        output / "class_safeguard_estimates.csv", index=False
    )
    metrics.to_csv(output / "class_safeguard_metrics.csv", index=False)
    aggregate.to_csv(output / "class_safeguard_aggregate.csv", index=False)
    bootstrap.to_csv(output / "class_safeguard_bootstrap.csv", index=False)
    with (output / "class_safeguard_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    return aggregate

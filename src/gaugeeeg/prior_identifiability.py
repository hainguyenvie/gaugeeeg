"""Cross-subject audit of operator-aware soft-confusion correction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from .bias_manifold import (
    _fit_ridge,
    _load_predictions,
    _montage_alias,
    _reference_name,
    _validate_view_grid,
    topology_descriptor,
)
from .calibration import fit_calibrator
from .prior_stress import (
    METHODS as E10_METHODS,
    BatchSelection,
    _apply_free_bias,
    _cluster_bootstrap_delta,
    _evaluate_target,
    _json_value,
    _make_batch_selections,
    _softmax,
    fit_known_prior_bias,
)


METHODS = (
    "identity",
    "topology_ridge",
    "uniform_prior_match",
    "fixed_topology_shrinkage",
    "operator_confusion_shrinkage",
    "oracle",
)
DYNAMIC_METHODS = (
    "uniform_prior_match",
    "fixed_topology_shrinkage",
    "operator_confusion_shrinkage",
)


@dataclass(frozen=True)
class SoftPriorModel:
    estimator: LogisticRegression
    soft_confusion: NDArray[np.float64]
    source_prior: NDArray[np.float64]
    oof_accuracy: float
    oof_balanced_accuracy: float
    condition_number: float


def _project_simplex(values: NDArray[np.floating]) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("Simplex projection requires a finite non-empty vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1)
    positive = ordered - cumulative / indices > 0.0
    if not positive.any():
        return np.full(vector.size, 1.0 / vector.size)
    rho = int(np.flatnonzero(positive)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    return projected / projected.sum()


def estimate_regularized_soft_prior(
    soft_confusion: NDArray[np.floating],
    observed_probability: NDArray[np.floating],
    nominal_prior: NDArray[np.floating] | list[float],
    *,
    regularization: float,
) -> NDArray[np.float64]:
    """Invert a soft confusion matrix with a nominal-prior ridge anchor."""

    confusion = np.asarray(soft_confusion, dtype=np.float64)
    observed = np.asarray(observed_probability, dtype=np.float64).reshape(-1)
    nominal = np.asarray(nominal_prior, dtype=np.float64).reshape(-1)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("soft_confusion must be a square matrix")
    if observed.size != confusion.shape[0] or nominal.size != confusion.shape[1]:
        raise ValueError("Soft-prior dimensions do not match")
    if not np.isfinite(confusion).all() or not np.isfinite(observed).all():
        raise ValueError("Soft-prior inputs must be finite")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if (nominal < 0.0).any() or not np.isclose(nominal.sum(), 1.0):
        raise ValueError("nominal_prior must be non-negative and sum to one")

    n_classes = nominal.size
    hessian = confusion.T @ confusion + regularization * np.eye(n_classes)
    target = confusion.T @ observed + regularization * nominal
    kkt = np.block(
        [
            [hessian, np.ones((n_classes, 1))],
            [np.ones((1, n_classes)), np.zeros((1, 1))],
        ]
    )
    right_hand_side = np.concatenate([target, np.asarray([1.0])])
    try:
        estimate = np.linalg.solve(kkt, right_hand_side)[:n_classes]
    except np.linalg.LinAlgError:
        estimate = np.linalg.lstsq(kkt, right_hand_side, rcond=None)[0][:n_classes]
    return _project_simplex(estimate)


def _center_logits(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    return values - values.mean(axis=1, keepdims=True)


def _balanced_accuracy(labels: NDArray[np.integer], prediction: NDArray[np.integer]) -> float:
    classes = np.unique(labels)
    return float(
        np.mean([np.mean(prediction[labels == value] == value) for value in classes])
    )


def _fit_soft_prior_model(
    frame: pd.DataFrame,
    logit_columns: list[str],
    subjects: list[int],
    *,
    seed: int,
) -> SoftPriorModel:
    selected = frame.loc[frame["subject_id"].isin(subjects)].copy()
    if selected.empty:
        raise ValueError("No rows were found for soft-prior model subjects")
    labels = selected["y_true"].to_numpy(dtype=np.int64)
    logits = _center_logits(selected[logit_columns].to_numpy(dtype=np.float64))
    n_classes = len(logit_columns)
    if not np.array_equal(np.unique(labels), np.arange(n_classes)):
        raise ValueError("Soft-prior model subjects must contain every class")

    estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    estimator.fit(logits, labels)
    if not np.array_equal(estimator.classes_, np.arange(n_classes)):
        raise RuntimeError("Unexpected class order in the soft-prior estimator")

    subject_ids = selected["subject_id"].to_numpy(dtype=np.int64)
    oof_probability = np.empty((labels.size, n_classes), dtype=np.float64)
    for held_out_subject in sorted(set(int(value) for value in subjects)):
        train_mask = subject_ids != held_out_subject
        held_out_mask = ~train_mask
        if not held_out_mask.any():
            raise ValueError(f"Soft-prior fold {held_out_subject} is empty")
        if not np.array_equal(np.unique(labels[train_mask]), np.arange(n_classes)):
            raise ValueError(
                f"Soft-prior fold excluding subject {held_out_subject} misses a class"
            )
        fold = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        fold.fit(logits[train_mask], labels[train_mask])
        oof_probability[held_out_mask] = fold.predict_proba(logits[held_out_mask])

    soft_confusion = np.stack(
        [
            oof_probability[labels == class_index].mean(axis=0)
            for class_index in range(n_classes)
        ],
        axis=1,
    )
    prediction = oof_probability.argmax(axis=1)
    source_prior = np.bincount(labels, minlength=n_classes) / labels.size
    return SoftPriorModel(
        estimator=estimator,
        soft_confusion=soft_confusion,
        source_prior=source_prior,
        oof_accuracy=float(np.mean(prediction == labels)),
        oof_balanced_accuracy=_balanced_accuracy(labels, prediction),
        condition_number=float(np.linalg.cond(soft_confusion)),
    )


def _selection_rows(
    selections: list[BatchSelection],
    labels: NDArray[np.integer],
    canonical: pd.DataFrame,
    *,
    role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_classes = int(labels.max()) + 1
    for selection in selections:
        counts = np.bincount(labels[selection.positions], minlength=n_classes)
        rows.append(
            {
                "role": role,
                "condition": selection.condition,
                "batch_size": selection.batch_size,
                "repeat": selection.repeat,
                "construction_uses_labels": selection.construction_uses_labels,
                "class_counts": json.dumps(counts.tolist()),
                "trial_indices": json.dumps(
                    canonical.iloc[selection.positions]["trial_index"].astype(int).tolist()
                ),
            }
        )
    return rows


def _aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["condition", "batch_size", "method"], as_index=False)
        .agg(
            n_rows=("target_view", "size"),
            n_target_views=("target_view", "nunique"),
            n_repeats=("repeat", "nunique"),
            mean_target_bias_rmse=("target_bias_rmse", "mean"),
            p90_target_bias_rmse=(
                "target_bias_rmse", lambda values: values.quantile(0.90)
            ),
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


def _mean_record(
    metrics: pd.DataFrame,
    *,
    condition: str | None = None,
    condition_prefix: str | None = None,
    batch_size: int,
    method: str,
) -> dict[str, Any]:
    selected = metrics.loc[(metrics["batch_size"] == batch_size) & (metrics["method"] == method)]
    if condition is not None:
        selected = selected.loc[selected["condition"] == condition]
    if condition_prefix is not None:
        selected = selected.loc[selected["condition"].str.startswith(condition_prefix)]
    if selected.empty:
        raise RuntimeError(f"Missing metrics for {condition or condition_prefix}, {method}")
    return {
        "condition": condition if condition is not None else f"{condition_prefix}*",
        "batch_size": batch_size,
        "method": method,
        "n_rows": int(selected.shape[0]),
        "n_target_views": int(selected["target_view"].nunique()),
        "n_repeats": int(selected["repeat"].nunique()),
        "mean_target_bias_rmse": float(selected["target_bias_rmse"].mean()),
        "mean_balanced_accuracy": float(selected["balanced_accuracy"].mean()),
        "mean_max_class_recall_gap": float(
            selected["max_class_recall_gap_to_car"].mean()
        ),
        "mean_nll": float(selected["nll"].mean()),
        "mean_ece_15": float(selected["ece_15"].mean()),
    }


def analyze_prior_identifiability(
    validation_predictions: str | Path,
    output_dir: str | Path,
    *,
    topology_subjects: list[int],
    prior_model_subjects: list[int],
    adaptation_subjects: list[int],
    evaluation_subjects: list[int],
    batch_sizes: list[int],
    primary_batch_size: int = 32,
    stress_batch_size: int = 128,
    n_resamples: int = 20,
    source_seed: int = 20260716,
    adaptation_seed: int = 20260717,
    ridge_alpha: float = 1.0,
    l2: float = 1e-4,
    confusion_regularization: float = 1.0,
    weak_confusion_regularization: float = 0.1,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    max_primary_rmse_increase: float = 0.05,
    minimum_severe_rmse_reduction: float = 0.05,
    max_mean_bacc_loss: float = 0.01,
    max_mean_gap_increase: float = 0.01,
) -> pd.DataFrame:
    """Audit whether soft-confusion structure resolves E10 prior confounding."""

    subject_groups = {
        "topology": set(topology_subjects),
        "prior_model": set(prior_model_subjects),
        "adaptation": set(adaptation_subjects),
        "evaluation": set(evaluation_subjects),
    }
    if any(not values for values in subject_groups.values()):
        raise ValueError("All E11 subject groups must be non-empty")
    if subject_groups["prior_model"] & subject_groups["adaptation"]:
        raise ValueError("Prior-model and adaptation subjects must be disjoint")
    if subject_groups["evaluation"] & subject_groups["topology"]:
        raise ValueError("Evaluation and topology-training subjects must be disjoint")
    if not (
        subject_groups["prior_model"] | subject_groups["adaptation"]
    ) <= subject_groups["topology"]:
        raise ValueError("Topology subjects must contain prior-model and adaptation subjects")
    if ridge_alpha <= 0.0 or l2 < 0.0:
        raise ValueError("Invalid ridge or calibration regularization")
    if confusion_regularization <= 0.0 or weak_confusion_regularization < 0.0:
        raise ValueError("Invalid soft-confusion regularization")
    if bootstrap_resamples < 1 or not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("Invalid bootstrap settings")

    frame, class_names = _load_predictions(validation_predictions)
    frame = frame.loc[
        frame["test_view"].astype(str).str.casefold().str.startswith("native")
    ].copy()
    if frame.empty:
        raise ValueError("No native validation views were found")
    logit_columns = [f"logit_{name}" for name in class_names]
    _validate_view_grid(frame, logit_columns)
    requested_subjects = set().union(*subject_groups.values())
    observed_subjects = set(frame["subject_id"].astype(int))
    if not requested_subjects <= observed_subjects:
        raise ValueError(
            f"Requested subjects are missing: {sorted(requested_subjects - observed_subjects)}"
        )

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

    def split_by_view(subjects: list[int]) -> dict[str, pd.DataFrame]:
        selected = frame.loc[frame["subject_id"].isin(subjects)]
        return {
            view: selected.loc[selected["test_view"] == view]
            .sort_values("trial_index")
            .reset_index(drop=True)
            for view in views
        }

    topology_by_view = split_by_view(topology_subjects)
    prior_by_view = split_by_view(prior_model_subjects)
    adaptation_by_view = split_by_view(adaptation_subjects)
    evaluation_by_view = split_by_view(evaluation_subjects)
    canonical_prior = prior_by_view[car_by_montage[montages[0]]]
    canonical_adaptation = adaptation_by_view[car_by_montage[montages[0]]]
    prior_labels = canonical_prior["y_true"].to_numpy(dtype=np.int64)
    adaptation_labels = canonical_adaptation["y_true"].to_numpy(dtype=np.int64)
    n_classes = len(class_names)
    if not np.array_equal(np.unique(prior_labels), np.arange(n_classes)):
        raise ValueError("Prior-model subjects must contain every class")
    if not np.array_equal(np.unique(adaptation_labels), np.arange(n_classes)):
        raise ValueError("Adaptation subjects must contain every class")
    nominal_prior = np.full(n_classes, 1.0 / n_classes)

    source_selections = _make_batch_selections(
        prior_labels,
        batch_sizes=batch_sizes,
        primary_batch_size=primary_batch_size,
        stress_batch_size=stress_batch_size,
        n_resamples=n_resamples,
        seed=source_seed,
    )
    adaptation_selections = _make_batch_selections(
        adaptation_labels,
        batch_sizes=batch_sizes,
        primary_batch_size=primary_batch_size,
        stress_batch_size=stress_batch_size,
        n_resamples=n_resamples,
        seed=adaptation_seed,
    )
    random_source_by_size = {
        size: [
            selection
            for selection in source_selections
            if selection.condition == "random" and selection.batch_size == size
        ]
        for size in sorted(set(int(value) for value in batch_sizes))
    }

    topology_descriptor_by_view = {
        view: topology_descriptor(view)[0] for view in views
    }
    oracle_bias: dict[str, NDArray[np.float64]] = {}
    known_prior_full_bias: dict[str, NDArray[np.float64]] = {}
    for view in views:
        logits = topology_by_view[view][logit_columns].to_numpy(dtype=np.float64)
        labels = topology_by_view[view]["y_true"].to_numpy(dtype=np.int64)
        oracle = fit_calibrator(logits, labels, "bias", l2=l2)
        known = fit_known_prior_bias(logits, nominal_prior, l2=l2)
        if not oracle.success or not known.success:
            raise RuntimeError(f"Full bias optimization failed for {view}")
        oracle_bias[view] = oracle.parameters
        known_prior_full_bias[view] = known.parameters

    outer_topology: dict[str, NDArray[np.float64]] = {}
    for target in targets:
        held_out = _reference_name(target)
        training = [
            view
            for view in views
            if _reference_name(view) == "car" or _reference_name(view) != held_out
        ]
        outer_topology[target] = _fit_ridge(
            np.stack([topology_descriptor_by_view[view] for view in training]),
            np.stack([oracle_bias[view] for view in training]),
            topology_descriptor_by_view[target],
            alpha=ridge_alpha,
        )

    unique_references = sorted({_reference_name(view) for view in targets})
    nested_topology: dict[tuple[str, str], NDArray[np.float64]] = {}
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
            prediction = _fit_ridge(
                np.stack([topology_descriptor_by_view[view] for view in training]),
                np.stack([oracle_bias[view] for view in training]),
                topology_descriptor_by_view[source],
                alpha=ridge_alpha,
            )
            nested_topology[(outer_reference, source)] = prediction
            errors.extend(np.square(prediction - oracle_bias[source]).tolist())
        topology_mse_by_outer[outer_reference] = float(np.mean(errors))

    source_batch_bias: dict[tuple[int, int, str], NDArray[np.float64]] = {}
    for size, selections in random_source_by_size.items():
        for selection in selections:
            for view in views:
                logits = prior_by_view[view].iloc[selection.positions][
                    logit_columns
                ].to_numpy(dtype=np.float64)
                fitted = fit_known_prior_bias(logits, nominal_prior, l2=l2)
                if not fitted.success:
                    raise RuntimeError(
                        f"Source bias failed for {view}, n={size}, repeat={selection.repeat}"
                    )
                source_batch_bias[(size, selection.repeat, view)] = fitted.parameters

    weight_rows: list[dict[str, Any]] = []
    base_weights: dict[tuple[str, int], float] = {}
    for outer_reference in unique_references:
        source_views = [
            view for view in targets if _reference_name(view) != outer_reference
        ]
        topology_mse = topology_mse_by_outer[outer_reference]
        for size, selections in random_source_by_size.items():
            prior_errors = [
                float(
                    np.mean(
                        np.square(
                            source_batch_bias[(size, selection.repeat, view)]
                            - oracle_bias[view]
                        )
                    )
                )
                for selection in selections
                for view in source_views
            ]
            prior_mse = float(np.mean(prior_errors))
            denominator = topology_mse + prior_mse
            prior_weight = (
                0.5
                if denominator <= np.finfo(float).eps
                else topology_mse / denominator
            )
            base_weights[(outer_reference, size)] = float(prior_weight)
            weight_rows.append(
                {
                    "held_out_reference": outer_reference,
                    "batch_size": size,
                    "prior_match_mse_from_source_subjects": prior_mse,
                    "nested_topology_mse": topology_mse,
                    "prior_match_weight": prior_weight,
                    "topology_weight": 1.0 - prior_weight,
                    "n_source_views": len(source_views),
                    "n_random_source_resamples": len(selections),
                    "target_reference_labels_used": False,
                }
            )

    prior_models: dict[str, SoftPriorModel] = {}
    model_rows: list[dict[str, Any]] = []
    for montage in montages:
        car_view = car_by_montage[montage]
        model = _fit_soft_prior_model(
            prior_by_view[car_view],
            logit_columns,
            prior_model_subjects,
            seed=source_seed,
        )
        prior_models[montage] = model
        model_rows.append(
            {
                "montage": montage,
                "car_view": car_view,
                "n_subjects": len(prior_model_subjects),
                "n_trials": int(prior_by_view[car_view].shape[0]),
                "oof_accuracy": model.oof_accuracy,
                "oof_balanced_accuracy": model.oof_balanced_accuracy,
                "soft_confusion_condition_number": model.condition_number,
                "source_prior": json.dumps(model.source_prior.tolist()),
                "soft_confusion": json.dumps(model.soft_confusion.tolist()),
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
    estimate_rows: list[dict[str, Any]] = []
    for selection in adaptation_selections:
        true_counts = np.bincount(
            adaptation_labels[selection.positions], minlength=n_classes
        )
        true_prior = true_counts / true_counts.sum()
        for target in targets:
            montage = _montage_alias(target)
            held_out = _reference_name(target)
            anchor_view = car_by_montage[montage]
            target_eval = evaluation_by_view[target]
            anchor_eval = evaluation_by_view[anchor_view]
            batch_logits = adaptation_by_view[target].iloc[selection.positions][
                logit_columns
            ].to_numpy(dtype=np.float64)
            uniform_fit = fit_known_prior_bias(batch_logits, nominal_prior, l2=l2)
            if not uniform_fit.success:
                raise RuntimeError(
                    f"Target uniform bias failed for {target}, {selection.condition}, "
                    f"n={selection.batch_size}, repeat={selection.repeat}"
                )
            topology_bias = outer_topology[target]
            prior_weight = base_weights[(held_out, selection.batch_size)]
            fixed_bias = topology_bias + prior_weight * (
                uniform_fit.parameters - topology_bias
            )

            corrected = _apply_free_bias(batch_logits, topology_bias)
            soft_probability = prior_models[montage].estimator.predict_proba(
                _center_logits(corrected)
            )
            observed_probability = soft_probability.mean(axis=0)
            weak_prior = estimate_regularized_soft_prior(
                prior_models[montage].soft_confusion,
                observed_probability,
                nominal_prior,
                regularization=weak_confusion_regularization,
            )
            candidate_prior = estimate_regularized_soft_prior(
                prior_models[montage].soft_confusion,
                observed_probability,
                nominal_prior,
                regularization=confusion_regularization,
            )
            soft_fit = fit_known_prior_bias(batch_logits, observed_probability, l2=l2)
            weak_fit = fit_known_prior_bias(batch_logits, weak_prior, l2=l2)
            candidate_fit = fit_known_prior_bias(batch_logits, candidate_prior, l2=l2)
            if not soft_fit.success or not weak_fit.success or not candidate_fit.success:
                raise RuntimeError(
                    f"Soft-confusion bias failed for {target}, {selection.condition}, "
                    f"n={selection.batch_size}, repeat={selection.repeat}"
                )
            soft_bias = topology_bias + prior_weight * (
                soft_fit.parameters - topology_bias
            )
            weak_bias = topology_bias + prior_weight * (
                weak_fit.parameters - topology_bias
            )
            candidate_bias = topology_bias + prior_weight * (
                candidate_fit.parameters - topology_bias
            )

            estimate_rows.append(
                {
                    "condition": selection.condition,
                    "batch_size": selection.batch_size,
                    "repeat": selection.repeat,
                    "target_view": target,
                    "montage": montage,
                    "held_out_reference": held_out,
                    "stress_construction_uses_labels": selection.construction_uses_labels,
                    "true_batch_prior_audit_only": json.dumps(true_prior.tolist()),
                    "soft_mean_prior": json.dumps(observed_probability.tolist()),
                    "weak_confusion_prior": json.dumps(weak_prior.tolist()),
                    "regularized_confusion_prior": json.dumps(candidate_prior.tolist()),
                    "soft_mean_prior_rmse": float(
                        np.sqrt(np.mean(np.square(observed_probability - true_prior)))
                    ),
                    "weak_confusion_prior_rmse": float(
                        np.sqrt(np.mean(np.square(weak_prior - true_prior)))
                    ),
                    "regularized_confusion_prior_rmse": float(
                        np.sqrt(np.mean(np.square(candidate_prior - true_prior)))
                    ),
                    "uniform_bias_rmse": float(
                        np.sqrt(
                            np.mean(np.square(uniform_fit.parameters - oracle_bias[target]))
                        )
                    ),
                    "fixed_shrinkage_bias_rmse": float(
                        np.sqrt(np.mean(np.square(fixed_bias - oracle_bias[target])))
                    ),
                    "soft_mean_shrinkage_bias_rmse": float(
                        np.sqrt(np.mean(np.square(soft_bias - oracle_bias[target])))
                    ),
                    "weak_confusion_shrinkage_bias_rmse": float(
                        np.sqrt(np.mean(np.square(weak_bias - oracle_bias[target])))
                    ),
                    "candidate_bias_rmse": float(
                        np.sqrt(np.mean(np.square(candidate_bias - oracle_bias[target])))
                    ),
                    "prior_match_weight": prior_weight,
                    "candidate_bias": json.dumps(candidate_bias.tolist()),
                    "target_reference_labels_used_for_candidate": False,
                }
            )

            dynamic = {
                "uniform_prior_match": _evaluate_target(
                    target_eval,
                    anchor_eval,
                    logit_columns,
                    uniform_fit.parameters,
                    known_prior_full_bias[anchor_view],
                    oracle_bias[target],
                ),
                "fixed_topology_shrinkage": _evaluate_target(
                    target_eval,
                    anchor_eval,
                    logit_columns,
                    fixed_bias,
                    known_prior_full_bias[anchor_view],
                    oracle_bias[target],
                ),
                "operator_confusion_shrinkage": _evaluate_target(
                    target_eval,
                    anchor_eval,
                    logit_columns,
                    candidate_bias,
                    known_prior_full_bias[anchor_view],
                    oracle_bias[target],
                ),
            }
            methods_for_selection = (
                METHODS if selection.repeat == 0 else DYNAMIC_METHODS
            )
            for method in methods_for_selection:
                method_metrics = (
                    dynamic[method]
                    if method in dynamic
                    else static_metrics[(target, method)]
                )
                metric_rows.append(
                    {
                        "condition": selection.condition,
                        "batch_size": selection.batch_size,
                        "repeat": selection.repeat,
                        "stress_construction_uses_labels": (
                            selection.construction_uses_labels
                        ),
                        "target_view": target,
                        "montage": montage,
                        "held_out_reference": held_out,
                        "method": method,
                        "target_reference_labels_used": method == "oracle",
                        "target_unlabeled_logits_used": method in DYNAMIC_METHODS,
                        "prior_match_weight": (
                            prior_weight
                            if method
                            in {
                                "fixed_topology_shrinkage",
                                "operator_confusion_shrinkage",
                            }
                            else float("nan")
                        ),
                        **method_metrics,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    estimates = pd.DataFrame(estimate_rows)
    aggregate = _aggregate_metrics(metrics)
    ablation = (
        estimates.groupby(["condition", "batch_size"], as_index=False)
        .agg(
            n_rows=("target_view", "size"),
            n_target_views=("target_view", "nunique"),
            n_repeats=("repeat", "nunique"),
            mean_soft_prior_rmse=("soft_mean_prior_rmse", "mean"),
            mean_weak_confusion_prior_rmse=("weak_confusion_prior_rmse", "mean"),
            mean_regularized_confusion_prior_rmse=(
                "regularized_confusion_prior_rmse", "mean"
            ),
            mean_uniform_bias_rmse=("uniform_bias_rmse", "mean"),
            mean_fixed_shrinkage_bias_rmse=("fixed_shrinkage_bias_rmse", "mean"),
            mean_soft_mean_shrinkage_bias_rmse=(
                "soft_mean_shrinkage_bias_rmse", "mean"
            ),
            mean_weak_confusion_shrinkage_bias_rmse=(
                "weak_confusion_shrinkage_bias_rmse", "mean"
            ),
            mean_candidate_bias_rmse=("candidate_bias_rmse", "mean"),
        )
        .sort_values(["condition", "batch_size"])
        .reset_index(drop=True)
    )

    primary = metrics.loc[
        (metrics["condition"] == "random")
        & (metrics["batch_size"] == primary_batch_size)
    ]
    severe = metrics.loc[
        metrics["condition"].str.startswith("skew_0.7_")
        & (metrics["batch_size"] == stress_batch_size)
    ]
    bootstrap_rows: list[dict[str, Any]] = []
    for comparison_index, (name, selected) in enumerate(
        (("primary_random", primary), ("severe_skew", severe))
    ):
        for metric_index, metric in enumerate(
            ["target_bias_rmse", "balanced_accuracy", "max_class_recall_gap_to_car"]
        ):
            row = _cluster_bootstrap_delta(
                selected,
                candidate="operator_confusion_shrinkage",
                baseline="fixed_topology_shrinkage",
                metric=metric,
                n_resamples=bootstrap_resamples,
                confidence=bootstrap_confidence,
                seed=adaptation_seed + 10 * comparison_index + metric_index,
            )
            row["comparison"] = name
            bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)

    primary_fixed = _mean_record(
        metrics,
        condition="random",
        batch_size=primary_batch_size,
        method="fixed_topology_shrinkage",
    )
    primary_candidate = _mean_record(
        metrics,
        condition="random",
        batch_size=primary_batch_size,
        method="operator_confusion_shrinkage",
    )
    balanced_fixed = _mean_record(
        metrics,
        condition="balanced",
        batch_size=stress_batch_size,
        method="fixed_topology_shrinkage",
    )
    balanced_candidate = _mean_record(
        metrics,
        condition="balanced",
        batch_size=stress_batch_size,
        method="operator_confusion_shrinkage",
    )
    severe_fixed = _mean_record(
        metrics,
        condition_prefix="skew_0.7_",
        batch_size=stress_batch_size,
        method="fixed_topology_shrinkage",
    )
    severe_candidate = _mean_record(
        metrics,
        condition_prefix="skew_0.7_",
        batch_size=stress_batch_size,
        method="operator_confusion_shrinkage",
    )
    severe_topology = _mean_record(
        metrics,
        condition_prefix="skew_0.7_",
        batch_size=stress_batch_size,
        method="topology_ridge",
    )
    severe_rmse_reduction = 1.0 - (
        severe_candidate["mean_target_bias_rmse"]
        / max(severe_fixed["mean_target_bias_rmse"], np.finfo(float).eps)
    )
    primary_rmse_ratio = primary_candidate["mean_target_bias_rmse"] / max(
        primary_fixed["mean_target_bias_rmse"], np.finfo(float).eps
    )
    balanced_rmse_ratio = balanced_candidate["mean_target_bias_rmse"] / max(
        balanced_fixed["mean_target_bias_rmse"], np.finfo(float).eps
    )

    per_dominant_class: list[dict[str, Any]] = []
    all_severe_classes_improve = True
    for class_index in range(n_classes):
        condition = f"skew_0.7_class_{class_index}"
        fixed = _mean_record(
            metrics,
            condition=condition,
            batch_size=stress_batch_size,
            method="fixed_topology_shrinkage",
        )
        candidate = _mean_record(
            metrics,
            condition=condition,
            batch_size=stress_batch_size,
            method="operator_confusion_shrinkage",
        )
        delta = (
            candidate["mean_target_bias_rmse"] - fixed["mean_target_bias_rmse"]
        )
        all_severe_classes_improve &= delta < 0.0
        per_dominant_class.append(
            {
                "dominant_class": class_index,
                "class_name": class_names[class_index],
                "fixed_mean_bias_rmse": fixed["mean_target_bias_rmse"],
                "candidate_mean_bias_rmse": candidate["mean_target_bias_rmse"],
                "candidate_minus_fixed": delta,
            }
        )
    n_severe_classes_improved = sum(
        row["candidate_minus_fixed"] < 0.0 for row in per_dominant_class
    )
    worst_severe_class_absolute_increase = max(
        0.0,
        max(row["candidate_minus_fixed"] for row in per_dominant_class),
    )
    worst_severe_class_relative_increase = max(
        0.0,
        max(
            row["candidate_minus_fixed"]
            / max(row["fixed_mean_bias_rmse"], np.finfo(float).eps)
            for row in per_dominant_class
        ),
    )

    severe_bootstrap = bootstrap.loc[
        (bootstrap["comparison"] == "severe_skew")
        & (bootstrap["metric"] == "target_bias_rmse")
    ].iloc[0]
    primary_preserved = bool(
        primary_rmse_ratio <= 1.0 + max_primary_rmse_increase
        and primary_candidate["mean_balanced_accuracy"]
        >= primary_fixed["mean_balanced_accuracy"] - max_mean_bacc_loss
        and primary_candidate["mean_max_class_recall_gap"]
        <= primary_fixed["mean_max_class_recall_gap"] + max_mean_gap_increase
    )
    balanced_preserved = bool(
        balanced_rmse_ratio <= 1.0 + max_primary_rmse_increase
        and balanced_candidate["mean_balanced_accuracy"]
        >= balanced_fixed["mean_balanced_accuracy"] - max_mean_bacc_loss
        and balanced_candidate["mean_max_class_recall_gap"]
        <= balanced_fixed["mean_max_class_recall_gap"] + max_mean_gap_increase
    )
    mean_severe_robustness_supported = bool(
        severe_rmse_reduction >= minimum_severe_rmse_reduction
        and severe_candidate["mean_target_bias_rmse"]
        < severe_topology["mean_target_bias_rmse"]
        and severe_candidate["mean_balanced_accuracy"]
        >= severe_fixed["mean_balanced_accuracy"] - max_mean_bacc_loss
        and severe_candidate["mean_max_class_recall_gap"]
        <= severe_fixed["mean_max_class_recall_gap"] + max_mean_gap_increase
        and severe_bootstrap["ci_upper"] < 0.0
    )
    class_uniform_severe_robustness_supported = bool(
        mean_severe_robustness_supported and all_severe_classes_improve
    )
    strict_severe_improved = class_uniform_severe_robustness_supported
    candidate_supported = bool(
        primary_preserved and balanced_preserved and strict_severe_improved
    )

    severe_estimates = estimates.loc[
        estimates["condition"].str.startswith("skew_0.7_")
        & (estimates["batch_size"] == stress_batch_size)
    ]
    severe_prior_rmse = float(
        severe_estimates["regularized_confusion_prior_rmse"].mean()
    )
    class_prior_identifiable = bool(severe_prior_rmse <= 0.10)
    weak_mean_bias_rmse = float(
        severe_estimates["weak_confusion_shrinkage_bias_rmse"].mean()
    )
    regularization_needed = bool(
        weak_mean_bias_rmse > severe_candidate["mean_target_bias_rmse"]
    )

    if candidate_supported and not class_prior_identifiable:
        recommendation = (
            "develop_operator_regularized_distribution_correction_not_prior_recovery"
        )
    elif candidate_supported:
        recommendation = "develop_operator_conditioned_label_shift_correction"
    elif mean_severe_robustness_supported and not all_severe_classes_improve:
        recommendation = "add_class_conditional_safeguard_before_method_claim"
    elif mean_severe_robustness_supported:
        recommendation = "improve_nominal_preservation_before_method_training"
    else:
        recommendation = "frozen_logits_do_not_resolve_reference_prior_confounding"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "prior_identifiability_metrics.csv", index=False)
    aggregate.to_csv(output / "prior_identifiability_aggregate.csv", index=False)
    estimates.to_csv(output / "prior_identifiability_estimates.csv", index=False)
    ablation.to_csv(output / "prior_identifiability_ablation.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(
        output / "prior_identifiability_weights.csv", index=False
    )
    pd.DataFrame(model_rows).to_csv(
        output / "prior_identifiability_models.csv", index=False
    )
    pd.DataFrame(
        _selection_rows(
            source_selections,
            prior_labels,
            canonical_prior,
            role="source_weight_training",
        )
        + _selection_rows(
            adaptation_selections,
            adaptation_labels,
            canonical_adaptation,
            role="target_adaptation_audit",
        )
    ).to_csv(output / "prior_identifiability_selections.csv", index=False)
    bootstrap.to_csv(output / "prior_identifiability_bootstrap.csv", index=False)

    summary = {
        "stage": "E11 cross-subject prior-identifiability audit",
        "topology_subjects": sorted(subject_groups["topology"]),
        "prior_model_subjects": sorted(subject_groups["prior_model"]),
        "adaptation_subjects": sorted(subject_groups["adaptation"]),
        "evaluation_subjects": sorted(subject_groups["evaluation"]),
        "prior_model_adaptation_subjects_disjoint": not bool(
            subject_groups["prior_model"] & subject_groups["adaptation"]
        ),
        "topology_evaluation_subjects_disjoint": not bool(
            subject_groups["topology"] & subject_groups["evaluation"]
        ),
        "source_and_target_batch_seeds_disjoint": source_seed != adaptation_seed,
        "physionet_test_subjects_used": False,
        "n_views": len(views),
        "n_target_views": len(targets),
        "held_out_unit": "reference electrode identity across all montages",
        "class_names": class_names,
        "nominal_prior": nominal_prior.tolist(),
        "batch_sizes": sorted(set(int(value) for value in batch_sizes)),
        "primary_random_batch_size": primary_batch_size,
        "stress_batch_size": stress_batch_size,
        "batch_resamples": n_resamples,
        "stress_batch_labels_used_for_construction_and_audit_only": True,
        "target_reference_labels_used_for_candidate": False,
        "prior_model_uses_car_reference_only": True,
        "soft_confusion_uses_leave_one_subject_out_predictions": True,
        "confusion_regularization_predeclared": confusion_regularization,
        "weak_confusion_regularization_ablation": weak_confusion_regularization,
        "ridge_alpha_predeclared": ridge_alpha,
        "l2_predeclared": l2,
        "e10_method_names_reused": list(E10_METHODS),
        "primary_fixed": primary_fixed,
        "primary_candidate": primary_candidate,
        "primary_candidate_to_fixed_rmse_ratio": primary_rmse_ratio,
        "primary_nominal_preserved": primary_preserved,
        "balanced_fixed": balanced_fixed,
        "balanced_candidate": balanced_candidate,
        "balanced_candidate_to_fixed_rmse_ratio": balanced_rmse_ratio,
        "balanced_nominal_preserved": balanced_preserved,
        "severe_fixed": severe_fixed,
        "severe_candidate": severe_candidate,
        "severe_topology": severe_topology,
        "severe_candidate_rmse_reduction_vs_fixed": severe_rmse_reduction,
        "all_severe_dominant_classes_improve": bool(all_severe_classes_improve),
        "n_severe_dominant_classes_improved": n_severe_classes_improved,
        "n_severe_dominant_classes_total": n_classes,
        "worst_severe_dominant_class_absolute_rmse_increase": (
            worst_severe_class_absolute_increase
        ),
        "worst_severe_dominant_class_relative_rmse_increase": (
            worst_severe_class_relative_increase
        ),
        "severe_per_dominant_class": per_dominant_class,
        "mean_severe_robustness_supported": mean_severe_robustness_supported,
        "class_uniform_severe_robustness_supported": (
            class_uniform_severe_robustness_supported
        ),
        "strict_severe_skew_improved": strict_severe_improved,
        "severe_skew_improved": strict_severe_improved,
        "primary_and_severe_bootstrap": [
            {key: _json_value(value) for key, value in row.items()}
            for row in bootstrap.to_dict(orient="records")
        ],
        "operator_confusion_shrinkage_supported": candidate_supported,
        "severe_regularized_prior_mean_rmse_to_true_batch_prior": severe_prior_rmse,
        "class_prior_identifiable_from_frozen_logits": class_prior_identifiable,
        "soft_confusion_regularization_needed": regularization_needed,
        "selection_rule": (
            f"Preserve random n={primary_batch_size} and balanced n={stress_batch_size} "
            f"within {max_primary_rmse_increase:.0%} RMSE and task-metric tolerances; "
            f"at severe skew n={stress_batch_size}, reduce RMSE by >= "
            f"{minimum_severe_rmse_reduction:.0%} versus fixed E10 shrinkage, beat "
            "topology, improve every dominant-class direction, preserve BAcc/recall gap, "
            f"and obtain a paired {bootstrap_confidence:.0%} RMSE-delta interval below zero."
        ),
        "next_method_recommendation": recommendation,
    }
    with (output / "prior_identifiability_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    return aggregate

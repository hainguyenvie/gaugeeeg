"""Validation-only calibration controls for montage/reference shifts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import log_loss, recall_score, roc_auc_score

from .metrics import classification_metrics_from_predictions


METHODS = ("temperature", "bias", "vector")


@dataclass(frozen=True)
class FittedCalibrator:
    method: str
    parameters: NDArray[np.float64]
    n_classes: int
    objective: float
    success: bool
    message: str

    def transform(self, logits: NDArray[np.floating]) -> NDArray[np.float64]:
        values = np.asarray(logits, dtype=np.float64)
        if self.method == "identity":
            return values.copy()
        if self.method == "temperature":
            return values / np.exp(self.parameters[0])
        if self.method == "bias":
            return values + _zero_sum_bias(self.parameters, self.n_classes)
        if self.method == "vector":
            scales = np.exp(self.parameters[: self.n_classes])
            biases = _zero_sum_bias(self.parameters[self.n_classes :], self.n_classes)
            return values * scales + biases
        raise ValueError(f"Unknown calibration method: {self.method}")


def _zero_sum_bias(parameters: NDArray[np.floating], n_classes: int) -> NDArray[np.float64]:
    free = np.asarray(parameters, dtype=np.float64)
    if free.size != n_classes - 1:
        raise ValueError("Bias parameter count does not match the number of classes")
    return np.concatenate([free, [-free.sum()]])


def _softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_calibrator(
    logits: NDArray[np.floating],
    labels: NDArray[np.integer],
    method: str,
    *,
    l2: float = 1e-4,
) -> FittedCalibrator:
    """Fit a small multiclass calibrator by validation negative log likelihood."""

    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    method = method.casefold()
    if values.ndim != 2 or values.shape[0] != targets.size or targets.size == 0:
        raise ValueError("Calibration logits and labels must be aligned non-empty arrays")
    if not np.isfinite(values).all():
        raise ValueError("Calibration logits must be finite")
    n_classes = values.shape[1]
    if not np.array_equal(np.unique(targets), np.arange(n_classes)):
        raise ValueError("Calibration labels must contain every class")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    if method == "identity":
        probability = _softmax(values)
        objective = float(log_loss(targets, probability, labels=np.arange(n_classes)))
        return FittedCalibrator(method, np.empty(0), n_classes, objective, True, "not fitted")
    if method == "temperature":
        initial = np.zeros(1)
        bounds = [(-4.0, 4.0)]
    elif method == "bias":
        initial = np.zeros(n_classes - 1)
        bounds = [(-5.0, 5.0)] * initial.size
    elif method == "vector":
        initial = np.zeros(n_classes + n_classes - 1)
        bounds = [(-2.0, 2.0)] * n_classes + [(-5.0, 5.0)] * (n_classes - 1)
    else:
        raise ValueError(f"Unknown calibration method {method!r}; expected identity or {METHODS}")

    def objective(parameters: NDArray[np.float64]) -> float:
        candidate = FittedCalibrator(method, parameters, n_classes, 0.0, True, "")
        calibrated = candidate.transform(values)
        nll = np.mean(logsumexp(calibrated, axis=1) - calibrated[np.arange(targets.size), targets])
        return float(nll + l2 * np.dot(parameters, parameters))

    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"{method} calibration produced non-finite parameters")
    return FittedCalibrator(
        method,
        np.asarray(result.x, dtype=np.float64),
        n_classes,
        float(result.fun),
        bool(result.success),
        str(result.message),
    )


def expected_calibration_error(
    labels: NDArray[np.integer],
    probabilities: NDArray[np.floating],
    *,
    n_bins: int = 15,
) -> float:
    """Top-label multiclass ECE with fixed-width confidence bins."""

    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] != targets.size or n_bins < 2:
        raise ValueError("ECE inputs must be aligned and n_bins must be at least two")
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = targets.size
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == 0.0
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def _load_predictions(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    source = Path(path)
    frame = pd.read_csv(source)
    required = {"test_view", "trial_index", "subject_id", "y_true", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")
    logit_columns = [column for column in frame.columns if column.startswith("logit_")]
    if len(logit_columns) < 2:
        raise ValueError(f"{source} must contain raw logit_* columns from the E8 run")
    class_names = [column.removeprefix("logit_") for column in logit_columns]
    if frame[list(required)].isna().any().any() or frame[logit_columns].isna().any().any():
        raise ValueError(f"{source} contains missing calibration values")
    frame = frame.sort_values(["test_view", "trial_index"]).reset_index(drop=True)
    return frame, class_names


def _view_group(view: str) -> str:
    return view.casefold().split("@", maxsplit=1)[0]


def _evaluate(
    labels: NDArray[np.integer],
    logits: NDArray[np.floating],
    identity_prediction: NDArray[np.integer],
) -> tuple[dict[str, float], NDArray[np.int64], NDArray[np.float64]]:
    probability = _softmax(logits)
    prediction = probability.argmax(axis=1).astype(np.int64)
    metrics = classification_metrics_from_predictions(labels, prediction, probability)
    recalls = recall_score(labels, prediction, average=None, labels=np.arange(probability.shape[1]))
    metrics.update(
        {
            "nll": float(log_loss(labels, probability, labels=np.arange(probability.shape[1]))),
            "ece_15": expected_calibration_error(labels, probability, n_bins=15),
            "worst_class_recall": float(recalls.min()),
            "prediction_disagreement_from_identity": float(
                np.mean(prediction != identity_prediction)
            ),
        }
    )
    return metrics, prediction, probability


def _bootstrap_bacc_delta(
    labels: NDArray[np.integer],
    identity: NDArray[np.integer],
    calibrated: NDArray[np.integer],
    subjects: NDArray[np.integer],
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    from sklearn.metrics import balanced_accuracy_score

    unique_subjects = np.unique(subjects)
    subject_indices = {subject: np.flatnonzero(subjects == subject) for subject in unique_subjects}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.choice(unique_subjects, size=unique_subjects.size, replace=True)
        indices = np.concatenate([subject_indices[subject] for subject in sample])
        deltas[index] = balanced_accuracy_score(labels[indices], calibrated[indices]) - (
            balanced_accuracy_score(labels[indices], identity[indices])
        )
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_subjects": int(unique_subjects.size),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "point_bacc_delta": float(
            balanced_accuracy_score(labels, calibrated)
            - balanced_accuracy_score(labels, identity)
        ),
        "bootstrap_mean_bacc_delta": float(deltas.mean()),
        "ci_lower": float(np.quantile(deltas, alpha)),
        "ci_upper": float(np.quantile(deltas, 1.0 - alpha)),
        "probability_delta_positive": float(np.mean(deltas > 0.0)),
    }


def _confirm_baseline_reproduction(current: pd.DataFrame, baseline_path: str | Path) -> bool:
    baseline = pd.read_csv(Path(baseline_path))
    keys = ["test_view", "trial_index", "subject_id", "y_true"]
    needed = set(keys + ["y_pred"])
    if not needed.issubset(baseline.columns):
        raise ValueError("E7e baseline predictions are missing alignment columns")
    left = current[keys + ["y_pred"]].sort_values(keys).reset_index(drop=True)
    right = baseline[keys + ["y_pred"]].sort_values(keys).reset_index(drop=True)
    if left.shape != right.shape or not left[keys].equals(right[keys]):
        raise RuntimeError("E8 and E7e test predictions are not trial-aligned")
    return bool(np.array_equal(left["y_pred"].to_numpy(), right["y_pred"].to_numpy()))


def analyze_calibration_controls(
    validation_predictions: str | Path,
    test_predictions: str | Path,
    output_dir: str | Path,
    *,
    baseline_predictions: str | Path | None = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20260715,
    l2: float = 1e-4,
    min_recall_shift_reduction: float = 0.50,
    max_worst_bacc_loss: float = 0.01,
) -> pd.DataFrame:
    """Fit calibration controls on validation logits and evaluate held-out test trials."""

    if n_resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap settings")
    validation, validation_classes = _load_predictions(validation_predictions)
    test, test_classes = _load_predictions(test_predictions)
    if validation_classes != test_classes:
        raise ValueError("Validation and test class/logit columns differ")
    validation_subjects = set(validation["subject_id"].astype(int))
    test_subjects = set(test["subject_id"].astype(int))
    split_subjects_disjoint = validation_subjects.isdisjoint(test_subjects)
    if not split_subjects_disjoint:
        raise RuntimeError("Validation and test subjects overlap; calibration would leak test labels")
    validation_views = list(dict.fromkeys(validation["test_view"].astype(str)))
    test_views = list(dict.fromkeys(test["test_view"].astype(str)))
    if set(validation_views) != set(test_views):
        raise ValueError("Validation and test prediction files must contain the same views")

    logit_columns = [f"logit_{name}" for name in validation_classes]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str, str], np.ndarray] = {}

    for view_index, view in enumerate(test_views):
        validation_target = validation.loc[validation["test_view"] == view]
        test_target = test.loc[test["test_view"] == view]
        labels = test_target["y_true"].to_numpy(dtype=np.int64)
        subjects = test_target["subject_id"].to_numpy(dtype=np.int64)
        test_logits = test_target[logit_columns].to_numpy(dtype=np.float64)
        identity_prediction = test_logits.argmax(axis=1).astype(np.int64)
        if not np.array_equal(identity_prediction, test_target["y_pred"].to_numpy(dtype=np.int64)):
            raise RuntimeError(f"Saved logits do not reproduce y_pred for {view}")

        protocols: list[tuple[str, str, pd.DataFrame]] = [
            ("none", "identity", validation_target),
        ]
        for method in METHODS:
            protocols.append(("view_specific", method, validation_target))
        group = _view_group(view)
        held_out_fit = validation.loc[
            (validation["test_view"].map(_view_group) == group)
            & (validation["test_view"] != view)
        ]
        if not held_out_fit.empty:
            for method in METHODS:
                protocols.append(("leave_one_view_out", method, held_out_fit))

        for protocol, method, fit_frame in protocols:
            calibrator = fit_calibrator(
                fit_frame[logit_columns].to_numpy(dtype=np.float64),
                fit_frame["y_true"].to_numpy(dtype=np.int64),
                method,
                l2=l2,
            )
            calibrated_logits = calibrator.transform(test_logits)
            metrics, prediction, probability = _evaluate(
                labels,
                calibrated_logits,
                identity_prediction,
            )
            predictions[(view, protocol, method)] = prediction
            metric_rows.append(
                {
                    "test_view": view,
                    "protocol": protocol,
                    "method": method,
                    "fit_views": "|".join(sorted(set(fit_frame["test_view"].astype(str)))),
                    "n_validation": int(fit_frame.shape[0]),
                    "n_test": int(test_target.shape[0]),
                    **metrics,
                }
            )
            recalls = recall_score(
                labels,
                prediction,
                average=None,
                labels=np.arange(len(validation_classes)),
            )
            for class_index, class_name in enumerate(validation_classes):
                binary = (labels == class_index).astype(int)
                try:
                    auroc = float(roc_auc_score(binary, probability[:, class_index]))
                except ValueError:
                    auroc = float("nan")
                class_rows.append(
                    {
                        "test_view": view,
                        "protocol": protocol,
                        "method": method,
                        "class_index": class_index,
                        "class_name": class_name,
                        "recall": float(recalls[class_index]),
                        "predicted_fraction": float(np.mean(prediction == class_index)),
                        "auroc": auroc,
                    }
                )
            parameter_rows.append(
                {
                    "test_view": view,
                    "protocol": protocol,
                    "method": method,
                    "fit_views": "|".join(sorted(set(fit_frame["test_view"].astype(str)))),
                    "n_validation": int(fit_frame.shape[0]),
                    "objective": calibrator.objective,
                    "optimizer_success": calibrator.success,
                    "optimizer_message": calibrator.message,
                    "parameters": json.dumps(calibrator.parameters.tolist()),
                }
            )
            if method != "identity":
                bootstrap = _bootstrap_bacc_delta(
                    labels,
                    identity_prediction,
                    prediction,
                    subjects,
                    n_resamples=n_resamples,
                    confidence=confidence,
                    seed=bootstrap_seed + view_index,
                )
                bootstrap_rows.append(
                    {
                        "test_view": view,
                        "protocol": protocol,
                        "method": method,
                        **bootstrap,
                        "ci_excludes_zero": bool(
                            bootstrap["ci_lower"] > 0.0 or bootstrap["ci_upper"] < 0.0
                        ),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    classes = pd.DataFrame(class_rows)
    stability_rows: list[dict[str, Any]] = []
    native_targets = [view for view in test_views if "@" in view and not view.endswith("@car")]
    for protocol, method in metrics[["protocol", "method"]].drop_duplicates().itertuples(index=False):
        gaps: list[float] = []
        available_targets = 0
        for target in native_targets:
            base = f"{_view_group(target)}@car"
            base_rows = classes.loc[
                (classes["test_view"] == base)
                & (classes["protocol"] == protocol)
                & (classes["method"] == method)
            ].sort_values("class_index")
            target_rows = classes.loc[
                (classes["test_view"] == target)
                & (classes["protocol"] == protocol)
                & (classes["method"] == method)
            ].sort_values("class_index")
            if base_rows.empty or target_rows.empty:
                continue
            gaps.extend(np.abs(base_rows["recall"].to_numpy() - target_rows["recall"].to_numpy()))
            available_targets += 1
        suite = metrics.loc[
            metrics["test_view"].str.startswith("native")
            & (metrics["protocol"] == protocol)
            & (metrics["method"] == method)
        ]
        if gaps and not suite.empty:
            stability_rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "n_reference_targets": available_targets,
                    "worst_absolute_class_recall_gap": float(max(gaps)),
                    "worst_native_balanced_accuracy": float(suite["balanced_accuracy"].min()),
                    "mean_native_balanced_accuracy": float(suite["balanced_accuracy"].mean()),
                }
            )
    stability = pd.DataFrame(stability_rows)
    identity_stability = stability.loc[
        (stability["protocol"] == "none") & (stability["method"] == "identity")
    ].iloc[0]
    stability["recall_shift_reduction"] = 1.0 - (
        stability["worst_absolute_class_recall_gap"]
        / identity_stability["worst_absolute_class_recall_gap"]
    )
    stability["worst_bacc_change_from_identity"] = (
        stability["worst_native_balanced_accuracy"]
        - identity_stability["worst_native_balanced_accuracy"]
    )
    stability["passes_explanatory_gate"] = (
        (stability["recall_shift_reduction"] >= min_recall_shift_reduction)
        & (stability["worst_bacc_change_from_identity"] >= -max_worst_bacc_loss)
    )

    def primary_row(protocol: str) -> dict[str, Any] | None:
        candidates = stability.loc[
            (stability["protocol"] == protocol) & (stability["method"] == "bias")
        ]
        if candidates.empty:
            return None
        return {key: _json_value(value) for key, value in candidates.iloc[0].to_dict().items()}

    primary_view_specific = primary_row("view_specific")
    primary_loo = primary_row("leave_one_view_out")
    view_specific_passed = bool(
        primary_view_specific is not None and primary_view_specific["passes_explanatory_gate"]
    )
    loo_passed = bool(primary_loo is not None and primary_loo["passes_explanatory_gate"])
    if loo_passed:
        recommendation = "calibration_first_reference_generalization"
    elif view_specific_passed:
        recommendation = "view_specific_calibration_baseline_required"
    else:
        recommendation = "proceed_to_joint_representation_learning"

    baseline_reproduced = None
    if baseline_predictions is not None:
        baseline_reproduced = _confirm_baseline_reproduction(test, baseline_predictions)
        if not baseline_reproduced:
            raise RuntimeError("E8 identity predictions do not exactly reproduce E7e")

    metrics.to_csv(output / "calibration_metrics.csv", index=False)
    classes.to_csv(output / "calibration_class_metrics.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "calibration_parameters.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "calibration_bootstrap.csv", index=False)
    stability.to_csv(output / "calibration_stability.csv", index=False)
    summary = {
        "stage": "E8 validation-only calibration control",
        "class_names": validation_classes,
        "validation_subjects": sorted(validation_subjects),
        "test_subjects": sorted(test_subjects),
        "validation_test_subjects_disjoint": split_subjects_disjoint,
        "test_labels_used_for_fitting": False,
        "raw_logits_used": True,
        "temperature_argmax_invariant_control": True,
        "e7e_predictions_reproduced_exactly": baseline_reproduced,
        "protocols": ["view_specific", "leave_one_view_out"],
        "methods": list(METHODS),
        "primary_calibration_method": "bias",
        "selection_rule": (
            "Class-bias calibration is the predeclared primary method; temperature is an "
            "argmax-invariant control and vector scaling is a sensitivity analysis. The primary "
            "method passes only if worst class-recall shift falls by >= "
            f"{min_recall_shift_reduction:.0%} and worst native BAcc falls by no more than "
            f"{max_worst_bacc_loss:.3f}. No method is selected from test performance."
        ),
        "primary_view_specific": primary_view_specific,
        "primary_leave_one_view_out": primary_loo,
        "view_specific_calibration_explains_shift": view_specific_passed,
        "leave_one_view_out_calibration_generalizes": loo_passed,
        "next_method_recommendation": recommendation,
        "caveat": (
            "View-specific calibration is an oracle known-view baseline because it uses labeled "
            "validation trials from the target view; leave-one-view-out is the unseen-reference control."
        ),
    }
    with (output / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return metrics


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value

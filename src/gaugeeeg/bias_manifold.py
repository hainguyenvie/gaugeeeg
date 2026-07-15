"""Validation-only reference-bias manifold diagnostic."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import FittedCalibrator, expected_calibration_error, fit_calibrator
from .metrics import classification_metrics_from_predictions
from .montage import SPARSE_MONTAGES, parse_observation_view


SIMPLE_BASELINES = ("global_mean", "pooled_bias")
PREDICTED_METHODS = ("topology_ridge", "logit_ridge", "combined_ridge")
ALL_METHODS = ("identity", *SIMPLE_BASELINES, *PREDICTED_METHODS, "oracle")

_ANTERIOR_POSTERIOR = {
    "F": 1.0,
    "FC": 0.5,
    "C": 0.0,
    "CP": -0.5,
    "P": -1.0,
    "PO": -1.5,
    "O": -2.0,
}
_LATERAL_MAGNITUDE = {1: 0.25, 2: 0.25, 3: 0.55, 4: 0.55, 5: 0.85, 6: 0.85}


def nominal_topology_coordinate(channel_name: str) -> tuple[float, float]:
    """Map a 10--20 channel name to a deterministic nominal 2-D topology."""

    match = re.fullmatch(r"(FC|CP|PO|F|C|P|O)(z|[1-6])", channel_name.strip(), re.I)
    if match is None:
        raise ValueError(f"Unsupported nominal 10-20 channel name: {channel_name!r}")
    region, suffix = match.group(1).upper(), match.group(2).casefold()
    y = _ANTERIOR_POSTERIOR[region] / 2.0
    if suffix == "z":
        return 0.0, y
    number = int(suffix)
    magnitude = _LATERAL_MAGNITUDE[number]
    x = -magnitude if number % 2 else magnitude
    return x, y


def topology_descriptor(view: str) -> tuple[NDArray[np.float64], list[str]]:
    """Describe reference location relative to the retained montage centroid."""

    specification = parse_observation_view(view)
    if specification.montage not in SPARSE_MONTAGES:
        raise ValueError(f"Bias-manifold views must use a sparse native montage: {view}")
    channels = SPARSE_MONTAGES[specification.montage]
    coordinates = np.asarray([nominal_topology_coordinate(name) for name in channels])
    centroid = coordinates.mean(axis=0)
    reference = specification.reference.casefold()
    if reference == "car":
        reference_coordinate = centroid
        car_flag = 1.0
    else:
        normalized = {name.casefold(): name for name in channels}
        if reference not in normalized:
            raise ValueError(f"Reference {reference!r} is not retained by {specification.montage}")
        reference_coordinate = np.asarray(
            nominal_topology_coordinate(normalized[reference]),
            dtype=np.float64,
        )
        car_flag = 0.0
    delta = reference_coordinate - centroid
    x, y = reference_coordinate
    dx, dy = delta
    values = np.asarray(
        [
            x,
            y,
            centroid[0],
            centroid[1],
            dx,
            dy,
            np.linalg.norm(delta),
            abs(dx),
            abs(dy),
            dx * dy,
            dx**2,
            dy**2,
            math.log2(len(channels)) / 5.0,
            car_flag,
        ],
        dtype=np.float64,
    )
    names = [
        "reference_x",
        "reference_y",
        "centroid_x",
        "centroid_y",
        "relative_x",
        "relative_y",
        "centroid_distance",
        "absolute_relative_x",
        "absolute_relative_y",
        "relative_xy",
        "relative_x_squared",
        "relative_y_squared",
        "log_montage_size",
        "car_flag",
    ]
    return values, names


def _softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    return probability / probability.sum(axis=1, keepdims=True)


def logit_statistics_descriptor(
    logits: NDArray[np.floating],
) -> tuple[NDArray[np.float64], list[str]]:
    """Build a label-free descriptor from a batch of target-view logits."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Logit-statistics descriptor requires a non-empty 2-D array")
    centered = values - values.mean(axis=1, keepdims=True)
    probability = _softmax(values)
    prediction = probability.argmax(axis=1)
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1)
    confidence = probability.max(axis=1)
    fractions = np.asarray([np.mean(prediction == index) for index in range(values.shape[1])])
    descriptor = np.concatenate(
        [
            centered.mean(axis=0),
            centered.std(axis=0),
            fractions,
            [confidence.mean(), confidence.std(), entropy.mean(), entropy.std()],
        ]
    ).astype(np.float64)
    names = (
        [f"centered_logit_mean_{index}" for index in range(values.shape[1])]
        + [f"centered_logit_std_{index}" for index in range(values.shape[1])]
        + [f"predicted_fraction_{index}" for index in range(values.shape[1])]
        + ["confidence_mean", "confidence_std", "entropy_mean", "entropy_std"]
    )
    return descriptor, names


def _load_predictions(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    source = Path(path)
    frame = pd.read_csv(source)
    required = {"test_view", "trial_index", "subject_id", "y_true", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")
    logit_columns = [column for column in frame.columns if column.startswith("logit_")]
    if len(logit_columns) < 2:
        raise ValueError(f"{source} must contain raw logit_* columns")
    class_names = [column.removeprefix("logit_") for column in logit_columns]
    if frame[list(required)].isna().any().any() or frame[logit_columns].isna().any().any():
        raise ValueError(f"{source} contains missing prediction values")
    if frame.duplicated(["test_view", "trial_index"]).any():
        raise ValueError(f"{source} contains duplicate view/trial rows")
    return frame.sort_values(["test_view", "trial_index"]).reset_index(drop=True), class_names


def _reference_name(view: str) -> str:
    return parse_observation_view(view).reference.casefold()


def _montage_alias(view: str) -> str:
    return view.split("@", maxsplit=1)[0].casefold()


def _apply_bias(logits: NDArray[np.floating], free_bias: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    calibrator = FittedCalibrator(
        "bias",
        np.asarray(free_bias, dtype=np.float64),
        values.shape[1],
        float("nan"),
        True,
        "predicted",
    )
    return calibrator.transform(values)


def _fit_ridge(
    features: NDArray[np.floating],
    targets: NDArray[np.floating],
    target_features: NDArray[np.floating],
    *,
    alpha: float,
) -> NDArray[np.float64]:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64))
    return np.asarray(model.predict(np.asarray(target_features).reshape(1, -1))[0], dtype=np.float64)


def _evaluate_logits(
    labels: NDArray[np.integer],
    logits: NDArray[np.floating],
) -> tuple[dict[str, float], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    probability = _softmax(logits)
    prediction = probability.argmax(axis=1).astype(np.int64)
    recalls = recall_score(
        labels,
        prediction,
        average=None,
        labels=np.arange(probability.shape[1]),
    )
    metrics = classification_metrics_from_predictions(labels, prediction, probability)
    metrics.update(
        {
            "nll": float(log_loss(labels, probability, labels=np.arange(probability.shape[1]))),
            "ece_15": expected_calibration_error(labels, probability),
            "worst_class_recall": float(recalls.min()),
        }
    )
    return metrics, prediction, probability, recalls


def _confirm_e8_reproduction(
    current: pd.DataFrame,
    e8_path: str | Path,
    logit_columns: list[str],
) -> dict[str, Any]:
    previous, previous_classes = _load_predictions(e8_path)
    if [column.removeprefix("logit_") for column in logit_columns] != previous_classes:
        raise ValueError("E8 and E9 class columns differ")
    shared = sorted(set(current["test_view"]) & set(previous["test_view"]))
    if not shared:
        raise RuntimeError("E8 and E9 validation files have no shared views")
    keys = ["test_view", "trial_index", "subject_id", "y_true"]
    left = current.loc[current["test_view"].isin(shared), keys + ["y_pred", *logit_columns]]
    right = previous.loc[previous["test_view"].isin(shared), keys + ["y_pred", *logit_columns]]
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    aligned = left[keys].equals(right[keys])
    predictions_equal = aligned and np.array_equal(left["y_pred"], right["y_pred"])
    logits_equal = aligned and np.array_equal(
        left[logit_columns].to_numpy(),
        right[logit_columns].to_numpy(),
    )
    if not predictions_equal or not logits_equal:
        raise RuntimeError("E9 does not exactly reproduce shared E8 validation outputs")
    return {
        "shared_views": shared,
        "shared_trial_rows": int(left.shape[0]),
        "shared_predictions_reproduced_exactly": bool(predictions_equal),
        "shared_logits_reproduced_exactly": bool(logits_equal),
    }


def _validate_view_grid(frame: pd.DataFrame, logit_columns: list[str]) -> None:
    saved_prediction = frame["y_pred"].to_numpy(dtype=np.int64)
    logit_prediction = frame[logit_columns].to_numpy(dtype=np.float64).argmax(axis=1)
    if not np.array_equal(saved_prediction, logit_prediction):
        raise RuntimeError("Saved validation logits do not reproduce y_pred")

    # Do not inspect cross-view labels here: the leakage test deliberately
    # perturbs held-out-reference labels and candidate predictions must remain
    # unchanged. Trial and subject identity are sufficient to validate the grid.
    keys = ["trial_index", "subject_id"]
    for montage in sorted({_montage_alias(view) for view in frame["test_view"].unique()}):
        montage_views = sorted(
            view for view in frame["test_view"].unique() if _montage_alias(view) == montage
        )
        car_view = f"{montage}@car"
        if car_view not in montage_views:
            raise ValueError(f"Missing CAR anchor for {montage}")
        anchor = frame.loc[frame["test_view"] == car_view, keys].reset_index(drop=True)
        for view in montage_views:
            current = frame.loc[frame["test_view"] == view, keys].reset_index(drop=True)
            if not current.equals(anchor):
                raise RuntimeError(f"Validation trials are not aligned for {view} and {car_view}")


def analyze_bias_manifold(
    validation_predictions: str | Path,
    output_dir: str | Path,
    *,
    fit_subjects: list[int],
    evaluation_subjects: list[int],
    e8_validation_predictions: str | Path | None = None,
    ridge_alpha: float = 1.0,
    l2: float = 1e-4,
    minimum_rmse_reduction: float = 0.20,
    minimum_recall_gap_reduction: float = 0.30,
    max_mean_bacc_loss_vs_simple: float = 0.01,
) -> pd.DataFrame:
    """Predict oracle bias from topology and unlabeled logits without test labels."""

    if not fit_subjects or not evaluation_subjects:
        raise ValueError("fit_subjects and evaluation_subjects must be non-empty")
    if set(fit_subjects) & set(evaluation_subjects):
        raise ValueError("Bias-fit and evaluation subjects must be disjoint")
    if ridge_alpha <= 0.0:
        raise ValueError("ridge_alpha must be positive")

    frame, class_names = _load_predictions(validation_predictions)
    native_mask = frame["test_view"].astype(str).str.casefold().str.startswith("native")
    frame = frame.loc[native_mask].copy()
    if frame.empty:
        raise ValueError("No native validation views were found")
    observed_subjects = set(frame["subject_id"].astype(int))
    requested_subjects = set(fit_subjects) | set(evaluation_subjects)
    if not requested_subjects <= observed_subjects:
        raise ValueError(f"Requested subjects are missing: {sorted(requested_subjects - observed_subjects)}")
    logit_columns = [f"logit_{name}" for name in class_names]
    _validate_view_grid(frame, logit_columns)
    views = sorted(frame["test_view"].unique())
    car_views = {view for view in views if _reference_name(view) == "car"}
    montages = sorted({_montage_alias(view) for view in views})
    expected_car = {f"{montage}@car" for montage in montages}
    if car_views != expected_car:
        raise ValueError(f"Expected CAR anchors {sorted(expected_car)}, observed {sorted(car_views)}")

    fit_frame = frame.loc[frame["subject_id"].isin(fit_subjects)]
    evaluation_frame = frame.loc[frame["subject_id"].isin(evaluation_subjects)]
    oracle_bias: dict[str, np.ndarray] = {}
    topology: dict[str, np.ndarray] = {}
    logit_stats: dict[str, np.ndarray] = {}
    descriptor_rows: list[dict[str, Any]] = []
    topology_names: list[str] | None = None
    logit_names: list[str] | None = None
    for view in views:
        current = fit_frame.loc[fit_frame["test_view"] == view]
        calibrator = fit_calibrator(
            current[logit_columns].to_numpy(dtype=np.float64),
            current["y_true"].to_numpy(dtype=np.int64),
            "bias",
            l2=l2,
        )
        if not calibrator.success:
            raise RuntimeError(f"Oracle bias optimization failed for {view}: {calibrator.message}")
        oracle_bias[view] = calibrator.parameters
        topology[view], current_topology_names = topology_descriptor(view)
        logit_stats[view], current_logit_names = logit_statistics_descriptor(
            current[logit_columns].to_numpy(dtype=np.float64)
        )
        topology_names = topology_names or current_topology_names
        logit_names = logit_names or current_logit_names
        descriptor_rows.append(
            {
                "view": view,
                "montage": _montage_alias(view),
                "reference": _reference_name(view),
                "n_fit_trials": int(current.shape[0]),
                "oracle_bias_objective": calibrator.objective,
                "oracle_bias_optimizer_success": calibrator.success,
                "oracle_bias_parameters": json.dumps(calibrator.parameters.tolist()),
                **{
                    f"topology_{name}": value
                    for name, value in zip(current_topology_names, topology[view], strict=True)
                },
                **{
                    f"logit_{name}": value
                    for name, value in zip(current_logit_names, logit_stats[view], strict=True)
                },
            }
        )

    targets = [view for view in views if view not in car_views]
    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    for target in targets:
        held_out_reference = _reference_name(target)
        training_views = [
            view
            for view in views
            if _reference_name(view) == "car" or _reference_name(view) != held_out_reference
        ]
        if not training_views or target in training_views:
            raise RuntimeError("Leave-one-electrode-out construction failed")
        y_bias = np.stack([oracle_bias[view] for view in training_views])
        topology_x = np.stack([topology[view] for view in training_views])
        logit_x = np.stack([logit_stats[view] for view in training_views])
        combined_x = np.concatenate([topology_x, logit_x], axis=1)
        pooled_frame = fit_frame.loc[fit_frame["test_view"].isin(training_views)]
        pooled_calibrator = fit_calibrator(
            pooled_frame[logit_columns].to_numpy(dtype=np.float64),
            pooled_frame["y_true"].to_numpy(dtype=np.int64),
            "bias",
            l2=l2,
        )
        if not pooled_calibrator.success:
            raise RuntimeError(
                f"Pooled bias optimization failed while holding out {held_out_reference}: "
                f"{pooled_calibrator.message}"
            )
        predicted_bias = {
            "identity": np.zeros_like(oracle_bias[target]),
            "global_mean": y_bias.mean(axis=0),
            "pooled_bias": pooled_calibrator.parameters,
            "topology_ridge": _fit_ridge(
                topology_x,
                y_bias,
                topology[target],
                alpha=ridge_alpha,
            ),
            "logit_ridge": _fit_ridge(
                logit_x,
                y_bias,
                logit_stats[target],
                alpha=ridge_alpha,
            ),
            "combined_ridge": _fit_ridge(
                combined_x,
                y_bias,
                np.concatenate([topology[target], logit_stats[target]]),
                alpha=ridge_alpha,
            ),
            "oracle": oracle_bias[target],
        }
        predicted_bias = {
            method: np.clip(parameters, -5.0, 5.0)
            for method, parameters in predicted_bias.items()
        }

        target_eval = evaluation_frame.loc[evaluation_frame["test_view"] == target]
        car_view = f"{_montage_alias(target)}@car"
        car_eval = evaluation_frame.loc[evaluation_frame["test_view"] == car_view]
        keys = ["trial_index", "subject_id", "y_true"]
        if not target_eval[keys].reset_index(drop=True).equals(car_eval[keys].reset_index(drop=True)):
            raise RuntimeError(f"Evaluation trials are not aligned for {target} and {car_view}")
        labels = target_eval["y_true"].to_numpy(dtype=np.int64)
        identity_prediction = target_eval[logit_columns].to_numpy().argmax(axis=1)
        for method in ALL_METHODS:
            if method == "identity":
                target_logits = target_eval[logit_columns].to_numpy(dtype=np.float64)
                car_logits = car_eval[logit_columns].to_numpy(dtype=np.float64)
            else:
                target_logits = _apply_bias(
                    target_eval[logit_columns].to_numpy(dtype=np.float64),
                    predicted_bias[method],
                )
                car_logits = _apply_bias(
                    car_eval[logit_columns].to_numpy(dtype=np.float64),
                    oracle_bias[car_view],
                )
            metrics, prediction, _, target_recalls = _evaluate_logits(labels, target_logits)
            car_metrics, _, _, car_recalls = _evaluate_logits(labels, car_logits)
            recall_gaps = np.abs(target_recalls - car_recalls)
            bias_rmse = float(np.sqrt(np.mean((predicted_bias[method] - oracle_bias[target]) ** 2)))
            metric_rows.append(
                {
                    "target_view": target,
                    "montage": _montage_alias(target),
                    "held_out_reference": held_out_reference,
                    "method": method,
                    "n_bias_fit_subjects": len(set(fit_frame["subject_id"])),
                    "n_evaluation_subjects": len(set(target_eval["subject_id"])),
                    "n_training_views": len(training_views),
                    "target_reference_labels_used": method == "oracle",
                    "target_bias_rmse": bias_rmse,
                    "car_anchor_balanced_accuracy": car_metrics["balanced_accuracy"],
                    "max_class_recall_gap_to_car": float(recall_gaps.max()),
                    "mean_class_recall_gap_to_car": float(recall_gaps.mean()),
                    "prediction_disagreement_from_identity": float(
                        np.mean(prediction != identity_prediction)
                    ),
                    **metrics,
                }
            )
            for class_index, class_name in enumerate(class_names):
                class_rows.append(
                    {
                        "target_view": target,
                        "montage": _montage_alias(target),
                        "held_out_reference": held_out_reference,
                        "method": method,
                        "class_index": class_index,
                        "class_name": class_name,
                        "car_recall": float(car_recalls[class_index]),
                        "target_recall": float(target_recalls[class_index]),
                        "absolute_recall_gap": float(recall_gaps[class_index]),
                    }
                )
            for parameter_index, (predicted, oracle) in enumerate(
                zip(predicted_bias[method], oracle_bias[target], strict=True)
            ):
                bias_rows.append(
                    {
                        "target_view": target,
                        "montage": _montage_alias(target),
                        "held_out_reference": held_out_reference,
                        "method": method,
                        "parameter_index": parameter_index,
                        "predicted_bias": float(predicted),
                        "oracle_bias": float(oracle),
                        "error": float(predicted - oracle),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        current = metrics.loc[metrics["method"] == method]
        aggregate_rows.append(
            {
                "method": method,
                "n_held_out_views": int(current.shape[0]),
                "mean_balanced_accuracy": float(current["balanced_accuracy"].mean()),
                "worst_balanced_accuracy": float(current["balanced_accuracy"].min()),
                "mean_target_bias_rmse": float(current["target_bias_rmse"].mean()),
                "worst_class_recall_gap_to_car": float(
                    current["max_class_recall_gap_to_car"].max()
                ),
                "mean_class_recall_gap_to_car": float(
                    current["mean_class_recall_gap_to_car"].mean()
                ),
                "mean_nll": float(current["nll"].mean()),
                "mean_ece_15": float(current["ece_15"].mean()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    identity = aggregate.loc[aggregate["method"] == "identity"].iloc[0]
    global_mean = aggregate.loc[aggregate["method"] == "global_mean"].iloc[0]
    pooled_bias = aggregate.loc[aggregate["method"] == "pooled_bias"].iloc[0]
    best_simple_rmse = min(
        float(global_mean["mean_target_bias_rmse"]),
        float(pooled_bias["mean_target_bias_rmse"]),
    )
    best_simple_bacc = max(
        float(global_mean["mean_balanced_accuracy"]),
        float(pooled_bias["mean_balanced_accuracy"]),
    )
    aggregate["bias_rmse_reduction_vs_global"] = 1.0 - (
        aggregate["mean_target_bias_rmse"]
        / max(float(global_mean["mean_target_bias_rmse"]), np.finfo(float).eps)
    )
    aggregate["bias_rmse_reduction_vs_pooled"] = 1.0 - (
        aggregate["mean_target_bias_rmse"]
        / max(float(pooled_bias["mean_target_bias_rmse"]), np.finfo(float).eps)
    )
    aggregate["bias_rmse_reduction_vs_best_simple"] = 1.0 - (
        aggregate["mean_target_bias_rmse"] / max(best_simple_rmse, np.finfo(float).eps)
    )
    aggregate["worst_recall_gap_reduction_vs_identity"] = 1.0 - (
        aggregate["worst_class_recall_gap_to_car"]
        / max(float(identity["worst_class_recall_gap_to_car"]), np.finfo(float).eps)
    )
    aggregate["mean_bacc_change_vs_global"] = (
        aggregate["mean_balanced_accuracy"] - global_mean["mean_balanced_accuracy"]
    )
    aggregate["mean_bacc_change_vs_identity"] = (
        aggregate["mean_balanced_accuracy"] - identity["mean_balanced_accuracy"]
    )
    aggregate["mean_bacc_change_vs_pooled"] = (
        aggregate["mean_balanced_accuracy"] - pooled_bias["mean_balanced_accuracy"]
    )
    aggregate["mean_bacc_change_vs_best_simple"] = (
        aggregate["mean_balanced_accuracy"] - best_simple_bacc
    )
    aggregate["passes_predictability_gate"] = (
        (aggregate["bias_rmse_reduction_vs_best_simple"] >= minimum_rmse_reduction)
        & (
            aggregate["worst_recall_gap_reduction_vs_identity"]
            >= minimum_recall_gap_reduction
        )
        & (aggregate["mean_bacc_change_vs_best_simple"] >= -max_mean_bacc_loss_vs_simple)
    )

    def aggregate_record(method: str) -> dict[str, Any]:
        row = aggregate.loc[aggregate["method"] == method].iloc[0]
        return {key: _json_value(value) for key, value in row.to_dict().items()}

    topology_result = aggregate_record("topology_ridge")
    logit_result = aggregate_record("logit_ridge")
    combined_result = aggregate_record("combined_ridge")
    topology_supported = bool(topology_result["passes_predictability_gate"])
    logit_supported = bool(logit_result["passes_predictability_gate"])
    combined_supported = bool(combined_result["passes_predictability_gate"])
    if combined_supported:
        recommendation = "build_operator_and_logit_conditioned_calibrator"
    elif topology_supported:
        recommendation = "build_operator_conditioned_calibrator"
    elif logit_supported:
        recommendation = "build_label_free_logit_adaptive_calibrator"
    else:
        recommendation = "simple_bias_manifold_not_predictable"

    reproduction = None
    if e8_validation_predictions is not None:
        reproduction = _confirm_e8_reproduction(frame, e8_validation_predictions, logit_columns)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "bias_manifold_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "bias_manifold_class_metrics.csv", index=False)
    pd.DataFrame(bias_rows).to_csv(output / "bias_manifold_predictions.csv", index=False)
    pd.DataFrame(descriptor_rows).to_csv(output / "bias_manifold_descriptors.csv", index=False)
    aggregate.to_csv(output / "bias_manifold_aggregate.csv", index=False)
    summary = {
        "stage": "E9 validation-only reference-bias manifold diagnostic",
        "fit_subjects": sorted(set(int(value) for value in fit_subjects)),
        "evaluation_subjects": sorted(set(int(value) for value in evaluation_subjects)),
        "fit_evaluation_subjects_disjoint": not bool(set(fit_subjects) & set(evaluation_subjects)),
        "physionet_test_subjects_used_by_manifold_analysis": False,
        "held_out_unit": "reference electrode identity across all montages",
        "n_views": len(views),
        "n_target_views": len(targets),
        "montages": montages,
        "class_names": class_names,
        "ridge_alpha_predeclared": ridge_alpha,
        "primary_method": "combined_ridge",
        "simple_baselines": list(SIMPLE_BASELINES),
        "candidate_methods": list(PREDICTED_METHODS),
        "pooled_bias_scope": "all non-held-out reference views across both native montages",
        "best_simple_bias_rmse": best_simple_rmse,
        "best_simple_mean_balanced_accuracy": best_simple_bacc,
        "held_out_target_labels_used_for_candidate_fitting": False,
        "held_out_target_labels_used_for_oracle_and_rmse_only": True,
        "labels_from_non_target_references_used_for_training": True,
        "source_car_labels_used_for_anchor_calibration": True,
        "evaluation_labels_used_for_metrics_and_gate_only": True,
        "topology_descriptor": topology_names,
        "unlabeled_logit_descriptor": logit_names,
        "selection_rule": (
            "A predicted bias manifold is supported when leave-one-electrode-out bias RMSE "
            f"improves over the better of global-mean and pooled bias by >= "
            f"{minimum_rmse_reduction:.0%}, worst recall-gap "
            f"improves over identity by >= {minimum_recall_gap_reduction:.0%}, and mean BAcc "
            f"is no more than {max_mean_bacc_loss_vs_simple:.3f} below the better simple "
            "baseline. No method is selected from PhysioNet test performance."
        ),
        "global_mean": aggregate_record("global_mean"),
        "pooled_bias": aggregate_record("pooled_bias"),
        "topology_ridge": topology_result,
        "logit_ridge": logit_result,
        "combined_ridge": combined_result,
        "oracle_upper_bound": aggregate_record("oracle"),
        "topology_conditioning_supported": topology_supported,
        "unlabeled_logit_conditioning_supported": logit_supported,
        "combined_conditioning_supported": combined_supported,
        "next_method_recommendation": recommendation,
        "e8_shared_output_reproduction": reproduction,
        "geometry_caveat": (
            "The topology descriptor is a deterministic nominal 10-20 layout, not calibrated "
            "subject-specific electrode coordinates."
        ),
    }
    with (output / "bias_manifold_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return aggregate


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value

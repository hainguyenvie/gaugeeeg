"""End-to-end reference-shift experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import dump_config
from .datasets import EEGDataset, dataset_fingerprint, load_physionet_mi
from .features import Encoder, build_encoder
from .metrics import (
    classification_metrics_from_predictions,
    fit_probe,
    paired_subject_bootstrap_bacc_gap,
    representation_metrics,
)
from .montage import observation_metadata, parse_observation_view, prepare_observation_view
from .referencing import common_average
from .reproducibility import run_provenance


def _feature_key(
    *,
    encoder_signature: str,
    dataset_signature: str,
    split_name: str,
    subjects: list[int],
    view: str,
    defense: str,
    seed: int,
) -> str:
    payload = json.dumps(
        {
            "feature_pipeline": "gaugeeeg-observation:v2",
            "encoder": encoder_signature,
            "dataset": dataset_signature,
            "split": split_name,
            "subjects": subjects,
            "view": view,
            "defense": defense,
            "seed": seed,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _defend(x_uv: np.ndarray, defense: str) -> np.ndarray:
    key = defense.casefold()
    if key == "none":
        return x_uv
    if key == "car_canonicalize":
        return common_average(x_uv)
    raise ValueError(f"Unknown defense: {defense}")


def _extract_features(
    dataset: EEGDataset,
    *,
    encoder: Encoder,
    dataset_signature: str,
    split_name: str,
    subject_ids: list[int],
    view: str,
    defense: str,
    seed: int,
    cache_dir: Path,
    force_recompute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    subset = dataset.subset(subject_ids)
    cache_key = _feature_key(
        encoder_signature=encoder.cache_signature,
        dataset_signature=dataset_signature,
        split_name=split_name,
        subjects=subject_ids,
        view=view,
        defense=defense,
        seed=seed,
    )
    cache_path = cache_dir / f"{cache_key}.npz"
    if cache_path.exists() and not force_recompute:
        with np.load(cache_path, allow_pickle=False) as cached:
            return cached["features"], cached["labels"]

    referenced, observed_channel_names = prepare_observation_view(
        subset.x_uv,
        subset.channel_names,
        view,
        seed=seed,
    )
    protected = _defend(referenced, defense)
    features = encoder.transform(protected, observed_channel_names, subset.sfreq)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=subset.y)
    return features, subset.y


def _validate_labels(name: str, y: np.ndarray, expected_classes: int) -> None:
    present = np.unique(y)
    expected = np.arange(expected_classes)
    if not np.array_equal(present, expected):
        raise RuntimeError(f"{name} split has labels {present.tolist()}, expected {expected.tolist()}")


def _drift_features(encoder: Encoder, features: np.ndarray) -> np.ndarray:
    if features.ndim == 2:
        return features
    summarize = getattr(encoder, "summarize_cached", None)
    if summarize is None:
        raise RuntimeError("Encoder emitted token features without a drift summarizer")
    return summarize(features)


def _predict_outputs(
    predictor,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Predict once while retaining raw logits when the probe exposes them."""

    predict_logits = getattr(predictor, "predict_logits", None)
    if predict_logits is not None:
        logits = np.asarray(predict_logits(features), dtype=np.float64)
        if logits.ndim != 2:
            raise RuntimeError(f"Expected two-dimensional logits, observed shape {logits.shape}")
        shifted = logits - logits.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        prediction = logits.argmax(axis=1).astype(np.int64)
        return prediction, probability, logits

    prediction = np.asarray(predictor.predict(features), dtype=np.int64)
    try:
        probability = np.asarray(predictor.predict_proba(features), dtype=np.float64)
    except (AttributeError, ValueError):
        probability = None
    return prediction, probability, None


def _prediction_frame(
    *,
    probe_seed: int,
    reference_seed: int,
    defense: str,
    split: str,
    view: str,
    subjects: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    logits: np.ndarray | None,
    label_names: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "probe_seed": probe_seed,
            "reference_seed": reference_seed,
            "defense": defense,
            "split": split,
            "test_view": view,
            "trial_index": np.arange(labels.size),
            "subject_id": subjects,
            "y_true": labels,
            "y_pred": predictions,
            "correct": predictions == labels,
        }
    )
    if probabilities is not None:
        for class_index, class_name in enumerate(label_names):
            frame[f"prob_{class_name}"] = probabilities[:, class_index]
    if logits is not None:
        for class_index, class_name in enumerate(label_names):
            frame[f"logit_{class_name}"] = logits[:, class_index]
    return frame


def run_experiment(config: dict[str, Any]) -> pd.DataFrame:
    legacy_seed = int(config.get("seed", 7))
    data_config = config["data"]
    experiment = config["experiment"]
    probe_seed = int(experiment.get("probe_seed", legacy_seed))
    reference_seed = int(experiment.get("reference_seed", legacy_seed))
    strict_determinism = bool(experiment.get("strict_determinism", False))
    output_dir = Path(experiment.get("output_dir", "outputs/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, output_dir / "resolved_config.yaml")

    if strict_determinism:
        from .torch_probe import configure_torch_determinism

        configure_torch_determinism(probe_seed, strict=True)

    force_recompute = bool(experiment.get("force_recompute", False))
    dataset = load_physionet_mi(data_config, force_recompute=force_recompute)
    data_signature = dataset_fingerprint(data_config)
    encoder = build_encoder(experiment)
    encoder_metadata = getattr(
        encoder,
        "metadata",
        {"encoder": encoder.name, "encoder_revision": encoder.cache_signature},
    )
    feature_cache = Path(experiment.get("feature_cache_dir", output_dir / "feature_cache"))
    provenance = run_provenance()

    splits = {
        "train": [int(v) for v in data_config["train_subjects"]],
        "val": [int(v) for v in data_config["val_subjects"]],
        "test": [int(v) for v in data_config["test_subjects"]],
    }
    if data_config.get("audit_subjects"):
        splits["audit"] = [int(v) for v in data_config["audit_subjects"]]
    prediction_split = "audit" if "audit" in splits else "val"
    validation_predictions_only = bool(
        experiment.get("validation_predictions_only", False)
    )
    if validation_predictions_only and not experiment.get(
        "save_validation_predictions", False
    ):
        raise ValueError(
            "validation_predictions_only requires save_validation_predictions"
        )
    expected_classes = len(dataset.label_names)
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    validation_prediction_frames: list[pd.DataFrame] = []
    subject_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    validation_only_rows: list[dict[str, Any]] = []
    train_view = str(experiment.get("train_view", "car"))
    training_views = [str(view) for view in experiment.get("training_views", [train_view])]
    if not training_views or training_views[0].casefold() != "car":
        raise ValueError("training_views must be non-empty with CAR as view zero")
    probe_objective = str(experiment.get("probe_objective", "car_only")).casefold()
    consistency_weight = float(experiment.get("consistency_weight", 0.0))
    views = [str(view) for view in experiment.get("test_views", ["car"])]
    validation_prediction_views = [
        str(view) for view in experiment.get("validation_prediction_views", views)
    ]
    if not validation_prediction_views:
        raise ValueError("validation_prediction_views must not be empty")
    defenses = [str(defense) for defense in experiment.get("defenses", ["none"])]
    if any(
        parse_observation_view(view).channel_policy == "remove"
        for view in [*views, *validation_prediction_views]
    ):
        if str(experiment.get("probe", "sklearn_logreg")).casefold() == "reve_token":
            raise ValueError(
                "Native channel-subset views change token count and cannot use the fixed-width "
                "reve_token flattening head; use REVE attention/mean pooling with sklearn_logreg "
                "or implement a variable-set probe."
            )

    for defense in defenses:
        print(f"\n=== Encoder={encoder.name} | defense={defense} ===")
        train_features: list[np.ndarray] = []
        val_features: list[np.ndarray] = []
        train_y: np.ndarray | None = None
        val_y: np.ndarray | None = None
        for training_view in training_views:
            current_train_x, current_train_y = _extract_features(
                dataset,
                encoder=encoder,
                dataset_signature=data_signature,
                split_name="train",
                subject_ids=splits["train"],
                view=training_view,
                defense=defense,
                seed=reference_seed,
                cache_dir=feature_cache,
                force_recompute=force_recompute,
            )
            current_val_x, current_val_y = _extract_features(
                dataset,
                encoder=encoder,
                dataset_signature=data_signature,
                split_name="val",
                subject_ids=splits["val"],
                view=training_view,
                defense=defense,
                seed=reference_seed,
                cache_dir=feature_cache,
                force_recompute=force_recompute,
            )
            if train_y is None:
                train_y = current_train_y
                val_y = current_val_y
            elif not np.array_equal(train_y, current_train_y) or not np.array_equal(
                val_y, current_val_y
            ):
                raise RuntimeError("Train/validation label order changed across reference views")
            train_features.append(current_train_x)
            val_features.append(current_val_x)

        if train_y is None or val_y is None:
            raise RuntimeError("No training features were extracted")
        probe_name = str(experiment.get("probe", "sklearn_logreg")).casefold()
        if len(training_views) == 1:
            train_x = train_features[0]
            val_x = val_features[0]
        elif probe_name == "reve_set":
            # Native montages have different token counts. The variable-set
            # probe consumes an aligned tuple rather than an impossible dense
            # stack, while every view still shares trial and label order.
            train_x = tuple(train_features)
            val_x = tuple(val_features)
        else:
            train_x = np.stack(train_features, axis=1)
            val_x = np.stack(val_features, axis=1)
        _validate_labels("train", train_y, expected_classes)
        _validate_labels("validation", val_y, expected_classes)
        selected_c = float("nan")
        selected_epoch = 0
        validation_consistency_loss = float("nan")
        validation_prediction_disagreement = float("nan")
        if probe_name == "sklearn_logreg":
            if len(training_views) != 1:
                raise ValueError("sklearn_logreg does not support aligned multi-view training")
            probe = fit_probe(
                train_x,
                train_y,
                val_x,
                val_y,
                c_grid=[float(c) for c in experiment.get("c_grid", [1.0])],
                seed=probe_seed,
            )
            predictor = probe.model
            selected_c = probe.selected_c
            validation_score = probe.validation_balanced_accuracy
        elif probe_name in {"reve_token", "reve_set"}:
            if probe_name == "reve_token":
                from .torch_probe import fit_reve_token_probe
            else:
                from .set_probe import fit_reve_set_probe

            initial_query = getattr(encoder, "pretrained_query", None)
            if initial_query is None:
                raise ValueError("probe: reve_token requires encoder: reve")
            common_probe_kwargs = {
                "initial_query": initial_query,
                "n_classes": expected_classes,
                "seed": probe_seed,
                "device": str(experiment.get("device", "auto")),
                "batch_size": int(experiment.get("probe_batch_size", 32)),
                "epochs": int(experiment.get("probe_epochs", 20)),
                "learning_rate": float(experiment.get("probe_learning_rate", 1e-4)),
                "weight_decay": float(experiment.get("probe_weight_decay", 1e-2)),
                "dropout": float(experiment.get("probe_dropout", 0.1)),
                "warmup_epochs": int(experiment.get("probe_warmup_epochs", 5)),
                "patience": int(experiment.get("probe_patience", 5)),
                "clip_grad": float(experiment.get("probe_clip_grad", 2.0)),
                "deterministic": strict_determinism,
            }
            if probe_name == "reve_token":
                probe = fit_reve_token_probe(
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    objective=probe_objective,
                    consistency_weight=consistency_weight,
                    **common_probe_kwargs,
                )
            else:
                probe = fit_reve_set_probe(
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    n_queries=int(experiment.get("set_queries", 8)),
                    n_heads=int(experiment.get("set_heads", 8)),
                    ff_multiplier=int(experiment.get("set_ff_multiplier", 2)),
                    objective=probe_objective,
                    consistency_weight=consistency_weight,
                    consistency_view_weights=experiment.get(
                        "consistency_view_weights"
                    ),
                    **common_probe_kwargs,
                )
            predictor = probe.model
            selected_epoch = probe.selected_epoch
            validation_score = probe.validation_balanced_accuracy
            validation_consistency_loss = probe.validation_consistency_loss
            validation_prediction_disagreement = probe.validation_prediction_disagreement
            pd.DataFrame(probe.history).to_csv(output_dir / f"probe_history_{defense}.csv", index=False)

            if bool(experiment.get("save_probe_checkpoint", True)):
                try:
                    import torch

                    torch.save(
                        {
                            "model_state_dict": predictor.module.state_dict(),
                            "selected_epoch": selected_epoch,
                            "validation_balanced_accuracy": validation_score,
                            "validation_consistency_loss": validation_consistency_loss,
                            "validation_prediction_disagreement": (
                                validation_prediction_disagreement
                            ),
                            "probe": probe_name,
                            "probe_seed": probe_seed,
                            "reference_seed": reference_seed,
                            "strict_determinism": strict_determinism,
                            "probe_objective": probe_objective,
                            "training_views": training_views,
                            "consistency_weight": consistency_weight,
                            "consistency_view_weights": experiment.get(
                                "consistency_view_weights"
                            ),
                            "set_queries": int(experiment.get("set_queries", 0)),
                            "set_heads": int(experiment.get("set_heads", 0)),
                        },
                        output_dir / f"probe_best_{defense}.pt",
                    )
                except ImportError:
                    pass
        else:
            raise ValueError(f"Unknown probe: {probe_name}")

        if bool(experiment.get("save_validation_predictions", False)):
            prediction_subject_ids = splits[prediction_split]
            prediction_dataset = dataset.subset(prediction_subject_ids)
            prediction_subjects = prediction_dataset.subjects
            prediction_labels = prediction_dataset.y
            prediction_cache_namespace = str(
                experiment.get(
                    "validation_prediction_cache_namespace",
                    prediction_split,
                )
            )
            if prediction_split == "val":
                validation_feature_by_view = {
                    training_view.casefold(): features
                    for training_view, features in zip(
                        training_views, val_features, strict=True
                    )
                }
                if not np.array_equal(prediction_labels, val_y):
                    raise RuntimeError(
                        "Validation prediction labels do not match probe validation"
                    )
            else:
                validation_feature_by_view = {}
            for view in validation_prediction_views:
                key = view.casefold()
                if key not in validation_feature_by_view:
                    features, labels = _extract_features(
                        dataset,
                        encoder=encoder,
                        dataset_signature=data_signature,
                        split_name=prediction_cache_namespace,
                        subject_ids=prediction_subject_ids,
                        view=view,
                        defense=defense,
                        seed=reference_seed,
                        cache_dir=feature_cache,
                        force_recompute=force_recompute,
                    )
                    if not np.array_equal(prediction_labels, labels):
                        raise RuntimeError(
                            "Prediction label order changed across reference views"
                        )
                    validation_feature_by_view[key] = features
                prediction, probability, logits = _predict_outputs(
                    predictor,
                    validation_feature_by_view[key],
                )
                validation_prediction_frames.append(
                    _prediction_frame(
                        probe_seed=probe_seed,
                        reference_seed=reference_seed,
                        defense=defense,
                        split=prediction_split,
                        view=view,
                        subjects=prediction_subjects,
                        labels=prediction_labels,
                        predictions=prediction,
                        probabilities=probability,
                        logits=logits,
                        label_names=dataset.label_names,
                    )
                )

            validation_only_rows.append(
                {
                    "probe_seed": probe_seed,
                    "reference_seed": reference_seed,
                    "defense": defense,
                    "probe_validation_balanced_accuracy": validation_score,
                    "selected_epoch": selected_epoch,
                    "prediction_split": prediction_split,
                    "n_prediction_trials": int(prediction_labels.size),
                }
            )

        if validation_predictions_only:
            continue

        test_features: dict[str, np.ndarray] = {}
        test_labels: np.ndarray | None = None
        for view in views:
            features, labels = _extract_features(
                dataset,
                encoder=encoder,
                dataset_signature=data_signature,
                split_name="test",
                subject_ids=splits["test"],
                view=view,
                defense=defense,
                seed=reference_seed,
                cache_dir=feature_cache,
                force_recompute=force_recompute,
            )
            _validate_labels("test", labels, expected_classes)
            if test_labels is None:
                test_labels = labels
            elif not np.array_equal(test_labels, labels):
                raise RuntimeError("Test label order changed across reference views")
            test_features[view.casefold()] = features

        if "car" not in test_features or test_labels is None:
            raise RuntimeError("A CAR test view is required")
        car_features = test_features["car"]
        test_subjects = dataset.subset(splits["test"]).subjects
        if test_subjects.size != test_labels.size:
            raise RuntimeError("Test subject metadata is not aligned with cached test labels")

        predictions: dict[str, np.ndarray] = {}
        probabilities: dict[str, np.ndarray | None] = {}
        logits_by_view: dict[str, np.ndarray | None] = {}
        task_metrics: dict[str, dict[str, float]] = {}
        for view in views:
            key = view.casefold()
            predictions[key], probabilities[key], logits_by_view[key] = _predict_outputs(
                predictor,
                test_features[key],
            )
            task_metrics[key] = classification_metrics_from_predictions(
                test_labels,
                predictions[key],
                probabilities[key],
            )
        car_metrics = task_metrics["car"]

        for view in views:
            key = view.casefold()
            metrics = task_metrics[key]
            view_metadata = observation_metadata(dataset.channel_names, view)
            drift = representation_metrics(
                _drift_features(encoder, car_features),
                _drift_features(encoder, test_features[key]),
            )
            row = {
                "seed": probe_seed,
                "probe_seed": probe_seed,
                "reference_seed": reference_seed,
                "encoder": encoder.name,
                "defense": defense,
                "train_view": train_view,
                "training_views": "|".join(training_views),
                "test_view": view,
                **view_metadata,
                "n_train": int(train_y.size),
                "n_val": int(val_y.size),
                "n_test": int(test_labels.size),
                "probe": probe_name,
                "probe_objective": probe_objective,
                "consistency_weight": consistency_weight,
                "set_queries": int(experiment.get("set_queries", 0)),
                "set_heads": int(experiment.get("set_heads", 0)),
                "selected_c": selected_c,
                "selected_epoch": selected_epoch,
                "validation_balanced_accuracy": validation_score,
                "validation_consistency_loss": validation_consistency_loss,
                "validation_prediction_disagreement": validation_prediction_disagreement,
                **metrics,
                **drift,
                "balanced_accuracy_gap_from_car": (
                    car_metrics["balanced_accuracy"] - metrics["balanced_accuracy"]
                ),
            }
            rows.append(row)

            if bool(experiment.get("save_predictions", False)):
                prediction_frames.append(
                    _prediction_frame(
                        probe_seed=probe_seed,
                        reference_seed=reference_seed,
                        defense=defense,
                        split="test",
                        view=view,
                        subjects=test_subjects,
                        labels=test_labels,
                        predictions=predictions[key],
                        probabilities=probabilities[key],
                        logits=logits_by_view[key],
                        label_names=dataset.label_names,
                    )
                )

            for subject in np.unique(test_subjects):
                mask = test_subjects == subject
                subject_metrics = classification_metrics_from_predictions(
                    test_labels[mask],
                    predictions[key][mask],
                    None if probabilities[key] is None else probabilities[key][mask],
                )
                car_subject_metrics = classification_metrics_from_predictions(
                    test_labels[mask],
                    predictions["car"][mask],
                    None if probabilities["car"] is None else probabilities["car"][mask],
                )
                subject_rows.append(
                    {
                        "probe_seed": probe_seed,
                        "reference_seed": reference_seed,
                        "defense": defense,
                        "test_view": view,
                        "subject_id": int(subject),
                        "n_trials": int(mask.sum()),
                        **subject_metrics,
                        "balanced_accuracy_gap_from_car": (
                            car_subject_metrics["balanced_accuracy"] - subject_metrics["balanced_accuracy"]
                        ),
                    }
                )

            bootstrap_resamples = int(experiment.get("bootstrap_resamples", 0))
            if key != "car" and bootstrap_resamples > 0:
                bootstrap = paired_subject_bootstrap_bacc_gap(
                    test_labels,
                    predictions["car"],
                    predictions[key],
                    test_subjects,
                    n_resamples=bootstrap_resamples,
                    confidence=float(experiment.get("bootstrap_confidence", 0.95)),
                    seed=int(experiment.get("bootstrap_seed", 20260713)),
                )
                bootstrap_rows.append(
                    {
                        "probe_seed": probe_seed,
                        "reference_seed": reference_seed,
                        "defense": defense,
                        "test_view": view,
                        **bootstrap,
                        "ci_excludes_zero": bool(bootstrap["ci_lower"] > 0.0),
                    }
                )

    if validation_predictions_only:
        output_predictions = pd.concat(
            validation_prediction_frames, ignore_index=True
        )
        output_predictions.to_csv(
            output_dir / "validation_predictions.csv", index=False
        )
        results = pd.DataFrame(validation_only_rows)
        results.to_csv(output_dir / "metrics.csv", index=False)
        subject_sets = {name: set(values) for name, values in splits.items()}
        pairwise_disjoint = all(
            not subject_sets[left] & subject_sets[right]
            for index, left in enumerate(subject_sets)
            for right in list(subject_sets)[index + 1 :]
        )
        summary = {
            "encoder": encoder.name,
            "seed": probe_seed,
            "probe_seed": probe_seed,
            "reference_seed": reference_seed,
            "strict_determinism": strict_determinism,
            "probe": probe_name,
            "probe_objective": probe_objective,
            "set_queries": int(experiment.get("set_queries", 0)),
            "set_heads": int(experiment.get("set_heads", 0)),
            "training_views": training_views,
            "consistency_weight": consistency_weight,
            "consistency_view_weights": experiment.get(
                "consistency_view_weights"
            ),
            "validation_prediction_views": validation_prediction_views,
            "validation_predictions_only": True,
            "prediction_split": prediction_split,
            "validation_prediction_cache_namespace": str(
                experiment.get(
                    "validation_prediction_cache_namespace",
                    prediction_split,
                )
            ),
            "train_subjects": splits["train"],
            "probe_validation_subjects": splits["val"],
            "audit_subjects": splits[prediction_split],
            "reserved_test_subjects": splits["test"],
            "all_subject_splits_pairwise_disjoint": pairwise_disjoint,
            "probe_validation_audit_subjects_disjoint": not bool(
                subject_sets["val"] & subject_sets[prediction_split]
            ),
            "physionet_test_subjects_used_for_fitting_or_scoring": False,
            "validation_balanced_accuracy": float(
                validation_only_rows[0]["probe_validation_balanced_accuracy"]
            ),
            "selected_epoch": int(validation_only_rows[0]["selected_epoch"]),
            "n_prediction_trials": int(
                validation_only_rows[0]["n_prediction_trials"]
            ),
            "n_channels": len(dataset.channel_names),
            "sfreq": dataset.sfreq,
            "feature_cache_dir": str(feature_cache),
            "dataset_fingerprint": data_signature,
            "encoder_metadata": encoder_metadata,
            "provenance": provenance,
            "metrics_path": str(output_dir / "metrics.csv"),
            "validation_predictions_path": str(
                output_dir / "validation_predictions.csv"
            ),
        }
        with (output_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2)
        return results

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "metrics.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output_dir / "predictions.csv", index=False
        )
    if validation_prediction_frames:
        pd.concat(validation_prediction_frames, ignore_index=True).to_csv(
            output_dir / "validation_predictions.csv", index=False
        )
    if subject_rows:
        pd.DataFrame(subject_rows).to_csv(output_dir / "subject_metrics.csv", index=False)
    if bootstrap_rows:
        pd.DataFrame(bootstrap_rows).to_csv(output_dir / "paired_subject_bootstrap.csv", index=False)
    car_rows = results.loc[results["test_view"].str.lower() == "car"]
    if car_rows.empty:
        raise RuntimeError("Clean gate requires a CAR test view")
    # Runs that sweep a defense without an undefended arm (e.g. defenses=[car_canonicalize])
    # have no "none" row, so score the gate against the defense the run actually used.
    undefended = car_rows.loc[car_rows["defense"] == "none"]
    clean_bacc = float(
        (undefended if not undefended.empty else car_rows)["balanced_accuracy"].iloc[0]
    )
    gate_threshold = float(experiment.get("clean_gate_min_balanced_accuracy", 0.0))
    non_car = results[results["test_view"].str.lower() != "car"]
    if non_car.empty:
        largest_drop = 0.0
        largest_absolute_change = 0.0
        worst_view = None
    else:
        worst_index = non_car["balanced_accuracy_gap_from_car"].idxmax()
        largest_drop = float(non_car.loc[worst_index, "balanced_accuracy_gap_from_car"])
        largest_absolute_change = float(non_car["balanced_accuracy_gap_from_car"].abs().max())
        worst_view = str(non_car.loc[worst_index, "test_view"])
    stress_threshold = float(experiment.get("stress_effect_min_balanced_accuracy_drop", 0.03))
    summary = {
        "encoder": encoder.name,
        "seed": probe_seed,
        "probe_seed": probe_seed,
        "reference_seed": reference_seed,
        "strict_determinism": strict_determinism,
        "probe_objective": probe_objective,
        "training_views": training_views,
        "test_observation_views": views,
        "validation_prediction_views": validation_prediction_views,
        "consistency_weight": consistency_weight,
        "validation_balanced_accuracy": validation_score,
        "validation_consistency_loss": validation_consistency_loss,
        "validation_prediction_disagreement": validation_prediction_disagreement,
        "n_trials": int(dataset.y.size),
        "n_channels": len(dataset.channel_names),
        "sfreq": dataset.sfreq,
        "largest_balanced_accuracy_gap": float(results["balanced_accuracy_gap_from_car"].max()),
        "lowest_paired_cosine": float(results["paired_cosine_to_car"].min()),
        "clean_car_balanced_accuracy": clean_bacc,
        "clean_gate_min_balanced_accuracy": gate_threshold,
        "clean_gate_passed": bool(clean_bacc >= gate_threshold),
        "worst_reference_view": worst_view,
        "largest_balanced_accuracy_drop": largest_drop,
        "largest_absolute_balanced_accuracy_change": largest_absolute_change,
        "stress_effect_min_balanced_accuracy_drop": stress_threshold,
        "stress_effect_detected": bool(largest_drop >= stress_threshold),
        "feature_cache_dir": str(feature_cache),
        "dataset_fingerprint": data_signature,
        "encoder_metadata": encoder_metadata,
        "provenance": provenance,
        "metrics_path": str(output_dir / "metrics.csv"),
        "predictions_path": str(output_dir / "predictions.csv") if prediction_frames else None,
        "validation_predictions_path": (
            str(output_dir / "validation_predictions.csv")
            if validation_prediction_frames
            else None
        ),
        "subject_metrics_path": str(output_dir / "subject_metrics.csv"),
        "paired_subject_bootstrap_path": (
            str(output_dir / "paired_subject_bootstrap.csv") if bootstrap_rows else None
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    columns = [
        "encoder",
        "defense",
        "test_view",
        "balanced_accuracy",
        "balanced_accuracy_gap_from_car",
        "paired_cosine_to_car",
        "linear_cka_to_car",
    ]
    print("\n" + results[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved results to {output_dir}")
    return results

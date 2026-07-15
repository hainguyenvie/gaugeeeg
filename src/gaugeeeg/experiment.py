"""End-to-end reference-shift experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import dump_config
from .datasets import EEGDataset, load_physionet_mi
from .features import Encoder, build_encoder
from .metrics import (
    classification_metrics_from_predictions,
    fit_probe,
    paired_subject_bootstrap_bacc_gap,
    representation_metrics,
)
from .montage import observation_metadata, parse_observation_view, prepare_observation_view
from .referencing import common_average


def _feature_key(
    *,
    encoder_signature: str,
    split_name: str,
    subjects: list[int],
    view: str,
    defense: str,
    seed: int,
) -> str:
    payload = json.dumps(
        {
            "encoder": encoder_signature,
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
    encoder = build_encoder(experiment)
    feature_cache = Path(experiment.get("feature_cache_dir", output_dir / "feature_cache"))

    splits = {
        "train": [int(v) for v in data_config["train_subjects"]],
        "val": [int(v) for v in data_config["val_subjects"]],
        "test": [int(v) for v in data_config["test_subjects"]],
    }
    expected_classes = len(dataset.label_names)
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    subject_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    train_view = str(experiment.get("train_view", "car"))
    training_views = [str(view) for view in experiment.get("training_views", [train_view])]
    if not training_views or training_views[0].casefold() != "car":
        raise ValueError("training_views must be non-empty with CAR as view zero")
    probe_objective = str(experiment.get("probe_objective", "car_only")).casefold()
    consistency_weight = float(experiment.get("consistency_weight", 0.0))
    views = [str(view) for view in experiment.get("test_views", ["car"])]
    defenses = [str(defense) for defense in experiment.get("defenses", ["none"])]
    if any(parse_observation_view(view).channel_policy == "remove" for view in views):
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
        if len(training_views) == 1:
            train_x = train_features[0]
            val_x = val_features[0]
        else:
            train_x = np.stack(train_features, axis=1)
            val_x = np.stack(val_features, axis=1)
        _validate_labels("train", train_y, expected_classes)
        _validate_labels("validation", val_y, expected_classes)
        probe_name = str(experiment.get("probe", "sklearn_logreg")).casefold()
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
                if probe_objective != "car_only" or len(training_views) != 1:
                    raise ValueError("probe: reve_set currently requires CAR-only single-view training")
                probe = fit_reve_set_probe(
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    n_queries=int(experiment.get("set_queries", 8)),
                    n_heads=int(experiment.get("set_heads", 8)),
                    ff_multiplier=int(experiment.get("set_ff_multiplier", 2)),
                    **common_probe_kwargs,
                )
            predictor = probe.model
            selected_epoch = probe.selected_epoch
            validation_score = probe.validation_balanced_accuracy
            validation_consistency_loss = probe.validation_consistency_loss
            validation_prediction_disagreement = probe.validation_prediction_disagreement
            pd.DataFrame(probe.history).to_csv(output_dir / f"probe_history_{defense}.csv", index=False)

            try:
                import torch

                torch.save(
                    {
                        "model_state_dict": predictor.module.state_dict(),
                        "selected_epoch": selected_epoch,
                        "validation_balanced_accuracy": validation_score,
                        "validation_consistency_loss": validation_consistency_loss,
                        "validation_prediction_disagreement": validation_prediction_disagreement,
                        "probe": probe_name,
                        "probe_seed": probe_seed,
                        "reference_seed": reference_seed,
                        "strict_determinism": strict_determinism,
                        "probe_objective": probe_objective,
                        "training_views": training_views,
                        "consistency_weight": consistency_weight,
                        "set_queries": int(experiment.get("set_queries", 0)),
                        "set_heads": int(experiment.get("set_heads", 0)),
                    },
                    output_dir / f"probe_best_{defense}.pt",
                )
            except ImportError:
                pass
        else:
            raise ValueError(f"Unknown probe: {probe_name}")

        test_features: dict[str, np.ndarray] = {}
        test_labels: np.ndarray | None = None
        for view in views:
            features, labels = _extract_features(
                dataset,
                encoder=encoder,
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
        task_metrics: dict[str, dict[str, float]] = {}
        for view in views:
            key = view.casefold()
            predictions[key] = predictor.predict(test_features[key])
            try:
                probabilities[key] = predictor.predict_proba(test_features[key])
            except (AttributeError, ValueError):
                probabilities[key] = None
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
                prediction_frame = pd.DataFrame(
                    {
                        "probe_seed": probe_seed,
                        "reference_seed": reference_seed,
                        "defense": defense,
                        "test_view": view,
                        "trial_index": np.arange(test_labels.size),
                        "subject_id": test_subjects,
                        "y_true": test_labels,
                        "y_pred": predictions[key],
                        "correct": predictions[key] == test_labels,
                    }
                )
                probability = probabilities[key]
                if probability is not None:
                    for class_index, class_name in enumerate(dataset.label_names):
                        prediction_frame[f"prob_{class_name}"] = probability[:, class_index]
                prediction_frames.append(prediction_frame)

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

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "metrics.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output_dir / "predictions.csv", index=False
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
        "metrics_path": str(output_dir / "metrics.csv"),
        "predictions_path": str(output_dir / "predictions.csv") if prediction_frames else None,
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

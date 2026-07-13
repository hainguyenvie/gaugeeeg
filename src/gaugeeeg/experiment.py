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
from .metrics import classification_metrics, fit_probe, representation_metrics
from .referencing import apply_reference_view, common_average


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

    referenced = apply_reference_view(subset.x_uv, subset.channel_names, view, seed=seed)
    protected = _defend(referenced, defense)
    features = encoder.transform(protected, subset.channel_names, subset.sfreq)
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
    seed = int(config.get("seed", 7))
    data_config = config["data"]
    experiment = config["experiment"]
    output_dir = Path(experiment.get("output_dir", "outputs/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, output_dir / "resolved_config.yaml")

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
    train_view = str(experiment.get("train_view", "car"))
    views = [str(view) for view in experiment.get("test_views", ["car"])]
    defenses = [str(defense) for defense in experiment.get("defenses", ["none"])]

    for defense in defenses:
        print(f"\n=== Encoder={encoder.name} | defense={defense} ===")
        train_x, train_y = _extract_features(
            dataset,
            encoder=encoder,
            split_name="train",
            subject_ids=splits["train"],
            view=train_view,
            defense=defense,
            seed=seed,
            cache_dir=feature_cache,
            force_recompute=force_recompute,
        )
        val_x, val_y = _extract_features(
            dataset,
            encoder=encoder,
            split_name="val",
            subject_ids=splits["val"],
            view=train_view,
            defense=defense,
            seed=seed,
            cache_dir=feature_cache,
            force_recompute=force_recompute,
        )
        _validate_labels("train", train_y, expected_classes)
        _validate_labels("validation", val_y, expected_classes)
        probe_name = str(experiment.get("probe", "sklearn_logreg")).casefold()
        selected_c = float("nan")
        selected_epoch = 0
        if probe_name == "sklearn_logreg":
            probe = fit_probe(
                train_x,
                train_y,
                val_x,
                val_y,
                c_grid=[float(c) for c in experiment.get("c_grid", [1.0])],
                seed=seed,
            )
            predictor = probe.model
            selected_c = probe.selected_c
            validation_score = probe.validation_balanced_accuracy
        elif probe_name == "reve_token":
            from .torch_probe import fit_reve_token_probe

            initial_query = getattr(encoder, "pretrained_query", None)
            if initial_query is None:
                raise ValueError("probe: reve_token requires encoder: reve")
            probe = fit_reve_token_probe(
                train_x,
                train_y,
                val_x,
                val_y,
                initial_query=initial_query,
                n_classes=expected_classes,
                seed=seed,
                device=str(experiment.get("device", "auto")),
                batch_size=int(experiment.get("probe_batch_size", 32)),
                epochs=int(experiment.get("probe_epochs", 20)),
                learning_rate=float(experiment.get("probe_learning_rate", 1e-4)),
                weight_decay=float(experiment.get("probe_weight_decay", 1e-2)),
                dropout=float(experiment.get("probe_dropout", 0.1)),
                warmup_epochs=int(experiment.get("probe_warmup_epochs", 5)),
                patience=int(experiment.get("probe_patience", 5)),
                clip_grad=float(experiment.get("probe_clip_grad", 2.0)),
            )
            predictor = probe.model
            selected_epoch = probe.selected_epoch
            validation_score = probe.validation_balanced_accuracy
            pd.DataFrame(probe.history).to_csv(output_dir / f"probe_history_{defense}.csv", index=False)

            try:
                import torch

                torch.save(
                    {
                        "model_state_dict": predictor.module.state_dict(),
                        "selected_epoch": selected_epoch,
                        "validation_balanced_accuracy": validation_score,
                        "probe": probe_name,
                        "seed": seed,
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
                seed=seed,
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
        car_metrics = classification_metrics(predictor, car_features, test_labels)

        for view in views:
            key = view.casefold()
            metrics = classification_metrics(predictor, test_features[key], test_labels)
            drift = representation_metrics(
                _drift_features(encoder, car_features),
                _drift_features(encoder, test_features[key]),
            )
            row = {
                "seed": seed,
                "encoder": encoder.name,
                "defense": defense,
                "train_view": train_view,
                "test_view": view,
                "n_train": int(train_y.size),
                "n_val": int(val_y.size),
                "n_test": int(test_labels.size),
                "probe": probe_name,
                "selected_c": selected_c,
                "selected_epoch": selected_epoch,
                "validation_balanced_accuracy": validation_score,
                **metrics,
                **drift,
                "balanced_accuracy_gap_from_car": (
                    car_metrics["balanced_accuracy"] - metrics["balanced_accuracy"]
                ),
            }
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "metrics.csv", index=False)
    clean_bacc = float(
        results.loc[
            (results["defense"] == "none") & (results["test_view"].str.lower() == "car"),
            "balanced_accuracy",
        ].iloc[0]
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
        "seed": seed,
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

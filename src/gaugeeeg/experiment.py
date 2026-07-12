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
    encoder_name: str,
    split_name: str,
    subjects: list[int],
    view: str,
    defense: str,
    seed: int,
) -> str:
    payload = json.dumps(
        {
            "encoder": encoder_name,
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
        encoder_name=encoder.name,
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
    feature_cache = output_dir / "feature_cache"

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
        probe = fit_probe(
            train_x,
            train_y,
            val_x,
            val_y,
            c_grid=[float(c) for c in experiment.get("c_grid", [1.0])],
            seed=seed,
        )

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
        car_metrics = classification_metrics(probe.model, car_features, test_labels)

        for view in views:
            key = view.casefold()
            metrics = classification_metrics(probe.model, test_features[key], test_labels)
            drift = representation_metrics(car_features, test_features[key])
            row = {
                "seed": seed,
                "encoder": encoder.name,
                "defense": defense,
                "train_view": train_view,
                "test_view": view,
                "n_train": int(train_y.size),
                "n_val": int(val_y.size),
                "n_test": int(test_labels.size),
                "selected_c": probe.selected_c,
                "validation_balanced_accuracy": probe.validation_balanced_accuracy,
                **metrics,
                **drift,
                "balanced_accuracy_gap_from_car": (
                    car_metrics["balanced_accuracy"] - metrics["balanced_accuracy"]
                ),
            }
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "encoder": encoder.name,
        "seed": seed,
        "n_trials": int(dataset.y.size),
        "n_channels": len(dataset.channel_names),
        "sfreq": dataset.sfreq,
        "largest_balanced_accuracy_gap": float(results["balanced_accuracy_gap_from_car"].max()),
        "lowest_paired_cosine": float(results["paired_cosine_to_car"].min()),
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

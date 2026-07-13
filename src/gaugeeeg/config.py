"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must be a YAML mapping")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "experiment"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing mapping section: {section}")

    data = config["data"]
    splits = {
        "train": set(data.get("train_subjects", [])),
        "val": set(data.get("val_subjects", [])),
        "test": set(data.get("test_subjects", [])),
    }
    if not all(splits.values()):
        raise ValueError("Train, validation, and test subject lists must be non-empty")
    if splits["train"] & splits["val"] or splits["train"] & splits["test"] or splits["val"] & splits["test"]:
        raise ValueError("Subject splits must be disjoint")
    subjects = set().union(*splits.values())
    if min(subjects) < 1 or max(subjects) > 109:
        raise ValueError("PhysioNetMI subject IDs must be in [1, 109]")

    allowed_runs = {4, 6, 8, 10, 12, 14}
    runs = set(data.get("runs", []))
    if not runs or not runs <= allowed_runs:
        raise ValueError(f"Motor-imagery runs must be a non-empty subset of {sorted(allowed_runs)}")

    experiment = config["experiment"]
    if experiment.get("train_view", "car").lower() != "car":
        raise ValueError("v0.1 uses CAR as the clean training reference")
    if "car" not in [str(v).lower() for v in experiment.get("test_views", [])]:
        raise ValueError("test_views must include 'car' as the clean reference")
    probe = str(experiment.get("probe", "sklearn_logreg")).casefold()
    if probe not in {"sklearn_logreg", "reve_token"}:
        raise ValueError("probe must be 'sklearn_logreg' or 'reve_token'")
    if probe == "reve_token" and experiment.get("reve_pooling") != "tokens":
        raise ValueError("probe: reve_token requires reve_pooling: tokens")


def with_overrides(
    config: dict[str, Any],
    *,
    encoder: str | None = None,
    device: str | None = None,
    output_dir: str | None = None,
    force_recompute: bool | None = None,
) -> dict[str, Any]:
    result = deepcopy(config)
    experiment = result["experiment"]
    if encoder is not None:
        experiment["encoder"] = encoder
    if device is not None:
        experiment["device"] = device
    if output_dir is not None:
        experiment["output_dir"] = output_dir
    if force_recompute is not None:
        experiment["force_recompute"] = force_recompute
    validate_config(result)
    return result


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

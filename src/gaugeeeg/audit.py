"""Aggregate repeated statistical-audit runs without selecting a per-seed worst view."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def aggregate_audit_runs(run_dirs: list[str | Path], output_dir: str | Path) -> pd.DataFrame:
    if len(run_dirs) < 2:
        raise ValueError("At least two audit run directories are required")

    frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        frame = pd.read_csv(metrics_path)
        required = {
            "probe_seed",
            "reference_seed",
            "defense",
            "test_view",
            "balanced_accuracy",
            "balanced_accuracy_gap_from_car",
            "paired_cosine_to_car",
            "linear_cka_to_car",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{metrics_path} is missing columns: {missing}")
        frame["run_dir"] = str(run_dir)
        frames.append(frame)

        bootstrap_path = run_dir / "paired_subject_bootstrap.csv"
        if bootstrap_path.exists():
            bootstrap = pd.read_csv(bootstrap_path)
            bootstrap["run_dir"] = str(run_dir)
            bootstrap_frames.append(bootstrap)

    combined = pd.concat(frames, ignore_index=True)
    if combined["reference_seed"].nunique() != 1:
        raise ValueError("Audit runs must share one fixed reference_seed")

    group_columns = ["defense", "test_view"]
    aggregate = (
        combined.groupby(group_columns, as_index=False)
        .agg(
            n_probe_seeds=("probe_seed", "nunique"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            gap_from_car_mean=("balanced_accuracy_gap_from_car", "mean"),
            gap_from_car_std=("balanced_accuracy_gap_from_car", "std"),
            paired_cosine_mean=("paired_cosine_to_car", "mean"),
            linear_cka_mean=("linear_cka_to_car", "mean"),
        )
        .sort_values(group_columns)
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path / "all_metrics.csv", index=False)
    aggregate.to_csv(output_path / "aggregate_by_view.csv", index=False)
    if bootstrap_frames:
        pd.concat(bootstrap_frames, ignore_index=True).to_csv(
            output_path / "all_paired_subject_bootstrap.csv", index=False
        )

    shifted = aggregate[aggregate["test_view"].str.casefold() != "car"]
    worst = shifted.loc[shifted["gap_from_car_mean"].idxmax()]
    summary = {
        "probe_seeds": sorted(int(value) for value in combined["probe_seed"].unique()),
        "reference_seed": int(combined["reference_seed"].iloc[0]),
        "std_ddof": 1,
        "selection_rule": "largest mean gap for one fixed reference view across all probe seeds",
        "fixed_worst_reference_view": str(worst["test_view"]),
        "fixed_worst_reference_gap_mean": float(worst["gap_from_car_mean"]),
        "fixed_worst_reference_gap_std": float(worst["gap_from_car_std"]),
    }
    with (output_path / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return aggregate

"""Validation-only selection for the rule-consistency weight ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .method_compare import _run_effect, _run_seed, aggregate_consistency_methods


def _single_value(frame: pd.DataFrame, column: str, run_dir: Path) -> float:
    values = frame[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"Expected one {column} in {run_dir}, found {values.tolist()}")
    return float(values[0])


def _effective_lambda(metrics: pd.DataFrame, run_dir: Path) -> tuple[str, float]:
    objectives = metrics["probe_objective"].astype(str).str.casefold().unique()
    if len(objectives) != 1:
        raise ValueError(f"Expected one probe objective in {run_dir}")
    objective = str(objectives[0])
    if objective == "multi_view_ce":
        return objective, 0.0
    if objective != "rule_consistency":
        raise ValueError(f"Lambda ablation does not accept objective {objective!r} in {run_dir}")
    return objective, _single_value(metrics, "consistency_weight", run_dir)


def _optional_metric(metrics: pd.DataFrame, column: str) -> float:
    if column not in metrics:
        return float("nan")
    values = metrics[column].dropna().unique()
    return float(values[0]) if len(values) == 1 else float("nan")


def _validation_manifest(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Read only validation metadata; target-view predictions are not accessed."""

    rows = []
    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        metrics = pd.read_csv(run_dir / "metrics.csv")
        required = {
            "probe_seed",
            "probe_objective",
            "consistency_weight",
            "selected_epoch",
            "validation_balanced_accuracy",
        }
        missing = sorted(required - set(metrics.columns))
        if missing:
            raise ValueError(f"{run_dir / 'metrics.csv'} is missing columns: {missing}")
        objective, weight = _effective_lambda(metrics, run_dir)
        rows.append(
            {
                "probe_seed": _run_seed(run_dir),
                "lambda": weight,
                "probe_objective": objective,
                "run_dir": str(run_dir),
                "selected_epoch": int(_single_value(metrics, "selected_epoch", run_dir)),
                "validation_balanced_accuracy": _single_value(
                    metrics, "validation_balanced_accuracy", run_dir
                ),
                "validation_consistency_loss": _optional_metric(
                    metrics, "validation_consistency_loss"
                ),
                "validation_prediction_disagreement": _optional_metric(
                    metrics, "validation_prediction_disagreement"
                ),
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["lambda", "probe_seed"]).reset_index(drop=True)
    duplicates = manifest.duplicated(["probe_seed", "lambda"], keep=False)
    if duplicates.any():
        repeated = manifest.loc[duplicates, ["probe_seed", "lambda", "run_dir"]]
        raise ValueError(f"Duplicate seed/lambda runs:\n{repeated.to_string(index=False)}")
    return manifest


def _validate_grid(manifest: pd.DataFrame, expected_lambdas: list[float]) -> list[int]:
    seeds = sorted(int(value) for value in manifest["probe_seed"].unique())
    observed = sorted(float(value) for value in manifest["lambda"].unique())
    expected = sorted(float(value) for value in expected_lambdas)
    if len(observed) != len(expected) or not np.allclose(observed, expected, atol=1e-12):
        raise ValueError(f"Expected lambda grid {expected}, found {observed}")
    counts = manifest.groupby("lambda")["probe_seed"].nunique()
    incomplete = counts[counts != len(seeds)]
    if not incomplete.empty:
        raise ValueError(f"Every lambda must contain all seeds {seeds}: {incomplete.to_dict()}")
    return seeds


def _validation_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for weight, frame in manifest.groupby("lambda", sort=True):
        rows.append(
            {
                "lambda": float(weight),
                "n_probe_seeds": int(frame["probe_seed"].nunique()),
                "validation_balanced_accuracy_mean": float(
                    frame["validation_balanced_accuracy"].mean()
                ),
                "validation_balanced_accuracy_std": float(
                    frame["validation_balanced_accuracy"].std(ddof=1)
                ),
                "validation_consistency_loss_mean": float(
                    frame["validation_consistency_loss"].mean()
                ),
                "validation_consistency_loss_n": int(
                    frame["validation_consistency_loss"].notna().sum()
                ),
                "validation_prediction_disagreement_mean": float(
                    frame["validation_prediction_disagreement"].mean()
                ),
                "validation_prediction_disagreement_n": int(
                    frame["validation_prediction_disagreement"].notna().sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _test_ablation(
    manifest: pd.DataFrame, *, target_view: str, target_class: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, item in manifest.iterrows():
        effect = _run_effect(
            item["run_dir"],
            target_view=target_view,
            target_class=target_class,
        )
        rows.append(
            {
                "probe_seed": int(item["probe_seed"]),
                "lambda": float(item["lambda"]),
                "run_dir": item["run_dir"],
                **effect,
            }
        )
    by_seed = pd.DataFrame(rows).sort_values(["lambda", "probe_seed"])
    lambda_zero = by_seed[by_seed["lambda"] == 0.0].set_index("probe_seed")
    by_seed["target_class_recall_gap_recovery_vs_lambda0"] = by_seed.apply(
        lambda row: float(
            lambda_zero.loc[int(row["probe_seed"]), "target_class_recall_gap"]
            - row["target_class_recall_gap"]
        ),
        axis=1,
    )
    by_seed["clean_balanced_accuracy_gain_vs_lambda0"] = by_seed.apply(
        lambda row: float(
            row["car_balanced_accuracy"]
            - lambda_zero.loc[int(row["probe_seed"]), "car_balanced_accuracy"]
        ),
        axis=1,
    )

    numeric_columns = [
        "car_balanced_accuracy",
        "target_balanced_accuracy",
        "balanced_accuracy_gap",
        "target_class_recall_gap",
        "target_class_recall_gap_recovery_vs_lambda0",
        "clean_balanced_accuracy_gain_vs_lambda0",
    ]
    summary_rows = []
    for weight, frame in by_seed.groupby("lambda", sort=True):
        row: dict[str, float | int] = {
            "lambda": float(weight),
            "n_probe_seeds": int(frame["probe_seed"].nunique()),
        }
        for column in numeric_columns:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_std"] = float(frame[column].std(ddof=1))
        summary_rows.append(row)
    return by_seed, pd.DataFrame(summary_rows)


def analyze_lambda_ablation(
    baseline_dirs: list[str | Path],
    run_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    expected_lambdas: list[float] | None = None,
    target_view: str = "cz",
    target_class: int = 0,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20260714,
) -> pd.DataFrame:
    """Select lambda on validation only, then evaluate the frozen held-out view."""

    if expected_lambdas is None:
        expected_lambdas = [0.0, 0.3, 1.0, 3.0, 10.0]
    manifest = _validation_manifest(run_dirs)
    seeds = _validate_grid(manifest, expected_lambdas)
    baseline_by_seed = {_run_seed(path): str(path) for path in baseline_dirs}
    if sorted(baseline_by_seed) != seeds:
        raise ValueError(
            f"Baseline seeds {sorted(baseline_by_seed)} do not match ablation seeds {seeds}"
        )

    validation_summary = _validation_summary(manifest)
    ranked = validation_summary.sort_values(
        ["validation_balanced_accuracy_mean", "lambda"],
        ascending=[False, True],
    )
    selected_lambda = float(ranked.iloc[0]["lambda"])

    # Test predictions are accessed only after the validation-only choice above is frozen.
    test_by_seed, test_summary = _test_ablation(
        manifest,
        target_view=target_view,
        target_class=target_class,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path / "lambda_validation_by_seed.csv", index=False)
    validation_summary.to_csv(output_path / "lambda_validation_summary.csv", index=False)
    test_by_seed.to_csv(output_path / "lambda_test_ablation_by_seed.csv", index=False)
    test_summary.to_csv(output_path / "lambda_test_ablation_summary.csv", index=False)

    selected_evidence = None
    if selected_lambda > 0.0:
        augmentation_dirs = []
        selected_dirs = []
        ordered_baselines = []
        for seed in seeds:
            seed_rows = manifest[manifest["probe_seed"] == seed]
            augmentation_dirs.append(
                str(seed_rows[np.isclose(seed_rows["lambda"], 0.0)].iloc[0]["run_dir"])
            )
            selected_dirs.append(
                str(
                    seed_rows[np.isclose(seed_rows["lambda"], selected_lambda)].iloc[0][
                        "run_dir"
                    ]
                )
            )
            ordered_baselines.append(baseline_by_seed[seed])
        evidence_dir = output_path / "selected_vs_augmentation"
        aggregate_consistency_methods(
            ordered_baselines,
            augmentation_dirs,
            selected_dirs,
            evidence_dir,
            target_view=target_view,
            target_class=target_class,
            n_resamples=n_resamples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
        )
        selected_evidence = json.loads(
            (evidence_dir / "aggregate_method_summary.json").read_text(encoding="utf-8")
        )

    summary = {
        "probe_seeds": seeds,
        "candidate_lambdas": sorted(float(value) for value in expected_lambdas),
        "selection_metric": "mean validation balanced accuracy across CAR/Pz/FCz and probe seeds",
        "tie_break_rule": "choose the smaller lambda on an exact validation tie",
        "selection_uses_target_view": False,
        "held_out_target_view": target_view,
        "target_class_index": int(target_class),
        "selected_lambda": selected_lambda,
        "selected_lambda_is_rule_informed": bool(selected_lambda > 0.0),
        "selected_vs_augmentation": selected_evidence,
        "interpretation_rule": (
            "The selected lambda is fixed by validation before target-view predictions are read. "
            "All test-grid rows are transparent ablations and must not be used to reselect lambda."
        ),
    }
    with (output_path / "lambda_ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return validation_summary

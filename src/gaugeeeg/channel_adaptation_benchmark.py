"""Locked Phase-B audit for spherical-spline channel adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .baseline_benchmark import (
    EVALUATION_VIEWS,
    EXPECTED_AUDIT,
    EXPECTED_TRAIN,
    EXPECTED_VALIDATION,
    HISTORICALLY_INSPECTED_TEST,
    JOINT_TRAINING_VIEWS,
    BaselineSpec,
    _hierarchical_bacc_delta,
    _hierarchical_bacc_delta_views,
    _metric_row,
    _normalize_views,
    _validate_run,
)

CHANNEL_ADAPTATION_SPECS: dict[str, tuple[BaselineSpec, str]] = {
    "car_only": (BaselineSpec(("car",), "car_only"), "none"),
    "joint_multiview_ce": (
        BaselineSpec(JOINT_TRAINING_VIEWS, "multi_view_ce"),
        "none",
    ),
    "ssi_car_only": (
        BaselineSpec(("car",), "car_only"),
        "spherical_spline",
    ),
    "ssi_joint_multiview_ce": (
        BaselineSpec(JOINT_TRAINING_VIEWS, "multi_view_ce"),
        "spherical_spline",
    ),
}


def _parse_run_specs(run_specs: list[str]) -> dict[str, list[Path]]:
    grouped = {method: [] for method in CHANNEL_ADAPTATION_SPECS}
    for item in run_specs:
        if "=" not in item:
            raise ValueError(f"Invalid run specification {item!r}; expected method=path")
        method, raw_path = item.split("=", maxsplit=1)
        if method not in CHANNEL_ADAPTATION_SPECS:
            raise ValueError(
                f"Unknown Phase-B method {method!r}; expected {sorted(CHANNEL_ADAPTATION_SPECS)}"
            )
        grouped[method].append(Path(raw_path))
    missing = [method for method, paths in grouped.items() if not paths]
    if missing:
        raise ValueError(f"Missing required Phase-B runs: {missing}")
    return grouped


def analyze_channel_adaptation_benchmark(
    run_specs: list[str],
    output_dir: str | Path,
    *,
    expected_seeds: list[int] | tuple[int, ...] = (7, 21, 42),
    bootstrap_resamples: int = 10_000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260721,
) -> pd.DataFrame:
    """Validate Phase-B runs and compare SSI with matched Phase-A controls."""

    grouped = _parse_run_specs(run_specs)
    expected_seed_set = {int(value) for value in expected_seeds}
    frames: dict[str, dict[int, dict[str, pd.DataFrame]]] = {}
    metric_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    encoder_revisions: set[tuple[str, str]] = set()
    dataset_fingerprints: set[str] = set()

    for method, run_dirs in grouped.items():
        spec, defense = CHANNEL_ADAPTATION_SPECS[method]
        frames[method] = {}
        observed_seeds: set[int] = set()
        for run_dir in run_dirs:
            summary, by_view = _validate_run(
                method,
                run_dir,
                spec=spec,
                expected_defense=defense,
            )
            probe_seed = int(summary["probe_seed"])
            if probe_seed in observed_seeds:
                raise ValueError(f"Duplicate probe seed {probe_seed} for {method}")
            observed_seeds.add(probe_seed)
            frames[method][probe_seed] = by_view
            metadata = summary.get("encoder_metadata", {})
            encoder_revisions.add(
                (
                    str(metadata.get("model_revision")),
                    str(metadata.get("position_model_revision")),
                )
            )
            dataset_fingerprints.add(str(summary.get("dataset_fingerprint")))
            manifest_rows.append(
                {
                    "method": method,
                    "defense": defense,
                    "run_dir": str(run_dir),
                    "probe_seed": probe_seed,
                    "objective": spec.objective,
                    "training_views": "|".join(_normalize_views(spec.training_views)),
                    "dataset_fingerprint": summary.get("dataset_fingerprint"),
                    "model_revision": metadata.get("model_revision"),
                    "position_model_revision": metadata.get("position_model_revision"),
                    "historical_test_scored": False,
                }
            )
            metric_rows.extend(
                _metric_row(method, run_dir, probe_seed, view, by_view[view]) for view in EVALUATION_VIEWS
            )
        if observed_seeds != expected_seed_set:
            raise ValueError(
                f"{method} has probe seeds {sorted(observed_seeds)}, expected {sorted(expected_seed_set)}"
            )

    invalid_revisions = {"", "None", "unresolved"}
    if len(encoder_revisions) != 1 or any(
        revision in invalid_revisions for revision in next(iter(encoder_revisions), ())
    ):
        raise ValueError(f"Phase-B runs use invalid or different REVE revisions: {encoder_revisions}")
    if len(dataset_fingerprints) != 1 or "None" in dataset_fingerprints:
        raise ValueError(
            f"Phase-B runs use invalid or different dataset fingerprints: {dataset_fingerprints}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_seed = pd.DataFrame(metric_rows).sort_values(["method", "probe_seed", "test_view"])
    by_seed.to_csv(output / "channel_adaptation_metrics_by_seed.csv", index=False)
    pd.DataFrame(manifest_rows).sort_values(["method", "probe_seed"]).to_csv(
        output / "channel_adaptation_manifest.csv", index=False
    )
    aggregate = (
        by_seed.groupby(["method", "test_view"], as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            worst_class_recall_mean=("worst_class_recall", "mean"),
        )
        .sort_values(["method", "test_view"])
    )
    aggregate.to_csv(output / "channel_adaptation_metrics_summary.csv", index=False)

    native16_views = tuple(view for view in EVALUATION_VIEWS if view.startswith("native16@"))
    single_views = ("car", "native32@car", "native16@car", "native16@cz")
    comparisons = [(method, "car_only") for method in CHANNEL_ADAPTATION_SPECS if method != "car_only"]
    comparisons.append(("ssi_joint_multiview_ce", "joint_multiview_ce"))
    bootstrap_rows: list[dict[str, Any]] = []
    for comparison_index, (candidate, baseline) in enumerate(comparisons):
        for view_index, view in enumerate(single_views):
            result = _hierarchical_bacc_delta(
                {seed: frames[candidate][seed][view] for seed in expected_seed_set},
                {seed: frames[baseline][seed][view] for seed in expected_seed_set},
                n_resamples=bootstrap_resamples,
                confidence=bootstrap_confidence,
                seed=bootstrap_seed + comparison_index * 10 + view_index,
            )
            bootstrap_rows.append({"candidate": candidate, "baseline": baseline, "test_view": view, **result})
        result = _hierarchical_bacc_delta_views(
            frames[candidate],
            frames[baseline],
            views=native16_views,
            n_resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed + comparison_index * 10 + len(single_views),
        )
        bootstrap_rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "test_view": "native16_reference_mean",
                **result,
            }
        )
    pairwise = pd.DataFrame(bootstrap_rows)
    pairwise.to_csv(output / "channel_adaptation_pairwise_bootstrap.csv", index=False)

    control_clean = float(
        aggregate.loc[
            (aggregate["method"] == "car_only") & (aggregate["test_view"] == "car"),
            "balanced_accuracy_mean",
        ].iloc[0]
    )
    method_rows: list[dict[str, Any]] = []
    for method in CHANNEL_ADAPTATION_SPECS:
        selected = aggregate.loc[aggregate["method"] == method].set_index("test_view")
        clean = float(selected.loc["car", "balanced_accuracy_mean"])
        if method == "car_only":
            clean_ci_lower = clean_ci_upper = 0.0
        else:
            comparison = pairwise.loc[
                (pairwise["candidate"] == method)
                & (pairwise["baseline"] == "car_only")
                & (pairwise["test_view"] == "car")
            ].iloc[0]
            clean_ci_lower = float(comparison["ci_lower"])
            clean_ci_upper = float(comparison["ci_upper"])
        clean_delta = clean - control_clean
        method_rows.append(
            {
                "method": method,
                "defense": CHANNEL_ADAPTATION_SPECS[method][1],
                "clean_car_bacc": clean,
                "clean_delta_vs_car_only": clean_delta,
                "clean_delta_ci_lower": clean_ci_lower,
                "clean_delta_ci_upper": clean_ci_upper,
                "clean_noninferiority_passed": bool(clean_delta >= -0.01 and clean_ci_lower >= -0.01),
                "native32_car_bacc": float(selected.loc["native32@car", "balanced_accuracy_mean"]),
                "native16_car_bacc": float(selected.loc["native16@car", "balanced_accuracy_mean"]),
                "native16_reference_bacc_mean": float(
                    selected.loc[list(native16_views), "balanced_accuracy_mean"].mean()
                ),
                "native16_worst_class_recall": float(
                    selected.loc[list(native16_views), "worst_class_recall_mean"].min()
                ),
                "suite_bacc_mean": float(selected["balanced_accuracy_mean"].mean()),
                "suite_worst_view_bacc": float(selected["balanced_accuracy_mean"].min()),
            }
        )
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(output / "channel_adaptation_method_summary.csv", index=False)
    eligible = method_summary.loc[method_summary["clean_noninferiority_passed"]].sort_values(
        ["native16_reference_bacc_mean", "suite_bacc_mean"], ascending=False
    )
    strongest = str(eligible.iloc[0]["method"]) if not eligible.empty else None

    summary = {
        "stage": "GaugeEEG Phase B frozen-REVE channel-adaptation benchmark",
        "status": "development_method_screen_only",
        "adapter_under_test": "spherical_spline_interpolation",
        "adapter_is_gaugeeeg_novel_method": False,
        "physionetmi_is_globally_untouched_test": False,
        "historically_inspected_subjects": list(HISTORICALLY_INSPECTED_TEST),
        "external_dataset_required_for_confirmation": True,
        "expected_probe_seeds": sorted(expected_seed_set),
        "train_subjects": list(EXPECTED_TRAIN),
        "probe_validation_subjects": list(EXPECTED_VALIDATION),
        "audit_subjects": list(EXPECTED_AUDIT),
        "evaluation_views": list(EVALUATION_VIEWS),
        "required_methods": list(CHANNEL_ADAPTATION_SPECS),
        "primary_selection_metric": "mean BAcc over native16@{CAR,Cz,Pz,Fz}",
        "clean_noninferiority_margin": 0.01,
        "strongest_development_method": strongest,
        "selection_is_paper_confirmation": False,
        "dataset_fingerprint": next(iter(dataset_fingerprints)),
        "encoder_revisions": list(next(iter(encoder_revisions))),
        "bootstrap_unit": "probe seed x subject, paired across views",
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_confidence": bootstrap_confidence,
    }
    with (output / "channel_adaptation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return method_summary

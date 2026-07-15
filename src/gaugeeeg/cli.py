"""Command-line interface for GaugeEEG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config, with_overrides
from .synthetic import run_checks


def _run_command(args: argparse.Namespace) -> None:
    from .experiment import run_experiment

    config = load_config(args.config)
    config = with_overrides(
        config,
        encoder=args.encoder,
        device=args.device,
        output_dir=args.output_dir,
        force_recompute=True if args.force_recompute else None,
        probe_seed=args.probe_seed,
        reference_seed=args.reference_seed,
        probe_objective=args.probe_objective,
        consistency_weight=args.consistency_weight,
        training_views=args.training_views,
        test_views=args.test_views,
        defenses=args.defenses,
        set_queries=args.set_queries,
    )
    run_experiment(config)


def _download_command(args: argparse.Namespace) -> None:
    from .datasets import load_physionet_mi

    config = load_config(args.config)
    dataset = load_physionet_mi(config["data"], force_recompute=args.force_recompute)
    print(
        json.dumps(
            {
                "trials": int(dataset.y.size),
                "subjects": int(np_unique_count(dataset.subjects)),
                "channels": len(dataset.channel_names),
                "sfreq": dataset.sfreq,
                "shape": list(dataset.x_uv.shape),
            },
            indent=2,
        )
    )


def np_unique_count(values) -> int:
    import numpy as np

    return int(np.unique(values).size)


def _synthetic_command(args: argparse.Namespace) -> None:
    result = run_checks(args.seed)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def _summarize_command(args: argparse.Namespace) -> None:
    path = Path(args.metrics)
    frame = pd.read_csv(path)
    columns = [
        "encoder",
        "defense",
        "test_view",
        "balanced_accuracy",
        "balanced_accuracy_gap_from_car",
        "paired_cosine_to_car",
        "linear_cka_to_car",
    ]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    print(frame[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _aggregate_command(args: argparse.Namespace) -> None:
    from .audit import aggregate_audit_runs

    result = aggregate_audit_runs(args.runs, args.output_dir)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _class_bias_command(args: argparse.Namespace) -> None:
    from .class_bias import analyze_class_bias

    result = analyze_class_bias(
        args.runs,
        args.output_dir,
        n_resamples=args.bootstrap_resamples,
        confidence=args.bootstrap_confidence,
        seed=args.bootstrap_seed,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _compare_methods_command(args: argparse.Namespace) -> None:
    from .method_compare import compare_consistency_methods

    result = compare_consistency_methods(
        args.baseline,
        args.augmentation,
        args.consistency,
        args.output_dir,
        target_view=args.target_view,
        target_class=args.target_class,
        n_resamples=args.bootstrap_resamples,
        confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _aggregate_methods_command(args: argparse.Namespace) -> None:
    from .method_compare import aggregate_consistency_methods

    result = aggregate_consistency_methods(
        args.baselines,
        args.augmentations,
        args.consistencies,
        args.output_dir,
        target_view=args.target_view,
        target_class=args.target_class,
        n_resamples=args.bootstrap_resamples,
        confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _lambda_ablation_command(args: argparse.Namespace) -> None:
    from .lambda_ablation import analyze_lambda_ablation

    result = analyze_lambda_ablation(
        args.baselines,
        args.runs,
        args.output_dir,
        expected_lambdas=args.expected_lambdas,
        target_view=args.target_view,
        target_class=args.target_class,
        n_resamples=args.bootstrap_resamples,
        confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _montage_screen_command(args: argparse.Namespace) -> None:
    from .montage_screen import analyze_montage_screen

    result = analyze_montage_screen(
        args.car_only,
        args.canonical,
        args.augmentation,
        args.consistency,
        args.output_dir,
        primary_view=args.primary_view,
        target_class=args.target_class,
        selected_lambda=args.selected_lambda,
        n_resamples=args.bootstrap_resamples,
        confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _native_montage_screen_command(args: argparse.Namespace) -> None:
    from .native_montage_screen import analyze_native_montage_screen

    result = analyze_native_montage_screen(
        args.baseline,
        args.canonical,
        args.output_dir,
        primary_view=args.primary_view,
        n_resamples=args.bootstrap_resamples,
        confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _select_set_head_command(args: argparse.Namespace) -> None:
    from .set_head_selection import select_set_head

    result = select_set_head(
        args.runs,
        args.output_dir,
        expected_queries=args.expected_queries,
        clean_gate=args.clean_gate,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _reference_closure_command(args: argparse.Namespace) -> None:
    from .reference_closure import analyze_reference_closure

    result = analyze_reference_closure(
        args.full_run,
        args.native_run,
        args.selection,
        args.output_dir,
        n_resamples=args.bootstrap_resamples,
        confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _reference_geometry_command(args: argparse.Namespace) -> None:
    from .reference_geometry import analyze_reference_geometry

    result = analyze_reference_geometry(
        args.run,
        args.e7d_full_run,
        args.selection,
        args.output_dir,
        n_resamples=args.bootstrap_resamples,
        confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gaugeeeg", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the reference-shift experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--encoder", choices=["bandpower", "reve"])
    run.add_argument("--device", help="auto, cpu, cuda, or a torch device such as cuda:1")
    run.add_argument("--output-dir")
    run.add_argument("--force-recompute", action="store_true")
    run.add_argument("--probe-seed", type=int, help="Probe initialization/data-order seed")
    run.add_argument("--reference-seed", type=int, help="Fixed seed for stochastic reference views")
    run.add_argument(
        "--probe-objective",
        choices=["car_only", "multi_view_ce", "rule_consistency"],
    )
    run.add_argument("--consistency-weight", type=float)
    run.add_argument("--training-views", nargs="+", help="Override aligned train/validation views")
    run.add_argument("--test-views", nargs="+", help="Override evaluation observation views")
    run.add_argument("--defenses", nargs="+", help="Override preprocessing defenses")
    run.add_argument("--set-queries", type=int, help="Override learned queries for probe: reve_set")
    run.set_defaults(handler=_run_command)

    download = subparsers.add_parser("download", help="Download and cache the configured dataset")
    download.add_argument("--config", required=True)
    download.add_argument("--force-recompute", action="store_true")
    download.set_defaults(handler=_download_command)

    synthetic = subparsers.add_parser("synthetic", help="Run the algebraic invariance checks")
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.set_defaults(handler=_synthetic_command)

    summarize = subparsers.add_parser("summarize", help="Print the key columns from a metrics CSV")
    summarize.add_argument("--metrics", required=True)
    summarize.set_defaults(handler=_summarize_command)

    aggregate = subparsers.add_parser("aggregate-audit", help="Aggregate repeated probe-seed audits")
    aggregate.add_argument("--runs", nargs="+", required=True, help="Audit output directories")
    aggregate.add_argument("--output-dir", default="outputs/reve_statistical_audit_multiseed")
    aggregate.set_defaults(handler=_aggregate_command)

    class_bias = subparsers.add_parser(
        "class-bias-audit", help="Audit class-conditional shifts from saved predictions"
    )
    class_bias.add_argument("--runs", nargs="+", required=True)
    class_bias.add_argument("--output-dir", default="outputs/reve_class_bias_audit")
    class_bias.add_argument("--bootstrap-resamples", type=int, default=10000)
    class_bias.add_argument("--bootstrap-confidence", type=float, default=0.95)
    class_bias.add_argument("--bootstrap-seed", type=int, default=20260713)
    class_bias.set_defaults(handler=_class_bias_command)

    compare_methods = subparsers.add_parser(
        "compare-methods", help="Compare CAR-only, augmentation, and consistency probes"
    )
    compare_methods.add_argument("--baseline", required=True)
    compare_methods.add_argument("--augmentation", required=True)
    compare_methods.add_argument("--consistency", required=True)
    compare_methods.add_argument("--target-view", default="cz")
    compare_methods.add_argument("--target-class", type=int, default=0)
    compare_methods.add_argument("--output-dir", default="outputs/reve_consistency_comparison_s7")
    compare_methods.add_argument("--bootstrap-resamples", type=int, default=10000)
    compare_methods.add_argument("--bootstrap-confidence", type=float, default=0.95)
    compare_methods.add_argument("--bootstrap-seed", type=int, default=20260714)
    compare_methods.set_defaults(handler=_compare_methods_command)

    aggregate_methods = subparsers.add_parser(
        "aggregate-methods",
        help="Aggregate consistency comparisons across probe seeds",
    )
    aggregate_methods.add_argument("--baselines", nargs="+", required=True)
    aggregate_methods.add_argument("--augmentations", nargs="+", required=True)
    aggregate_methods.add_argument("--consistencies", nargs="+", required=True)
    aggregate_methods.add_argument("--target-view", default="cz")
    aggregate_methods.add_argument("--target-class", type=int, default=0)
    aggregate_methods.add_argument(
        "--output-dir", default="outputs/reve_consistency_comparison_multiseed"
    )
    aggregate_methods.add_argument("--bootstrap-resamples", type=int, default=10000)
    aggregate_methods.add_argument("--bootstrap-confidence", type=float, default=0.95)
    aggregate_methods.add_argument("--bootstrap-seed", type=int, default=20260714)
    aggregate_methods.set_defaults(handler=_aggregate_methods_command)

    lambda_ablation = subparsers.add_parser(
        "lambda-ablation",
        help="Select consistency weight on validation and audit held-out performance",
    )
    lambda_ablation.add_argument("--baselines", nargs="+", required=True)
    lambda_ablation.add_argument("--runs", nargs="+", required=True)
    lambda_ablation.add_argument(
        "--expected-lambdas",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 1.0, 3.0, 10.0],
    )
    lambda_ablation.add_argument("--target-view", default="cz")
    lambda_ablation.add_argument("--target-class", type=int, default=0)
    lambda_ablation.add_argument(
        "--output-dir", default="outputs/reve_consistency_lambda_ablation"
    )
    lambda_ablation.add_argument("--bootstrap-resamples", type=int, default=10000)
    lambda_ablation.add_argument("--bootstrap-confidence", type=float, default=0.95)
    lambda_ablation.add_argument("--bootstrap-seed", type=int, default=20260714)
    lambda_ablation.set_defaults(handler=_lambda_ablation_command)

    montage_screen = subparsers.add_parser(
        "montage-screen",
        help="Analyze the fixed reference-plus-sparse-montage feasibility screen",
    )
    montage_screen.add_argument("--car-only", required=True)
    montage_screen.add_argument("--canonical", required=True)
    montage_screen.add_argument("--augmentation", required=True)
    montage_screen.add_argument("--consistency", required=True)
    montage_screen.add_argument("--primary-view", default="sparse16@cz")
    montage_screen.add_argument("--target-class", type=int, default=0)
    montage_screen.add_argument("--selected-lambda", type=float, default=10.0)
    montage_screen.add_argument("--output-dir", default="outputs/reve_montage_screen_s7")
    montage_screen.add_argument("--bootstrap-resamples", type=int, default=10000)
    montage_screen.add_argument("--bootstrap-confidence", type=float, default=0.95)
    montage_screen.add_argument("--bootstrap-seed", type=int, default=20260714)
    montage_screen.set_defaults(handler=_montage_screen_command)

    native_screen = subparsers.add_parser(
        "native-montage-screen",
        help="Validate a native variable-channel REVE montage benchmark",
    )
    native_screen.add_argument("--baseline", required=True)
    native_screen.add_argument("--canonical", required=True)
    native_screen.add_argument("--primary-view", default="native16@cz")
    native_screen.add_argument("--output-dir", default="outputs/reve_native_montage_screen_s7")
    native_screen.add_argument("--bootstrap-resamples", type=int, default=10000)
    native_screen.add_argument("--bootstrap-confidence", type=float, default=0.95)
    native_screen.add_argument("--bootstrap-seed", type=int, default=20260714)
    native_screen.set_defaults(handler=_native_montage_screen_command)

    set_head = subparsers.add_parser(
        "select-set-head",
        help="Select the E7c variable-set query count using CAR validation only",
    )
    set_head.add_argument("--runs", nargs="+", required=True)
    set_head.add_argument("--expected-queries", nargs="+", type=int, default=[4, 8, 16])
    set_head.add_argument("--clean-gate", type=float, default=0.45)
    set_head.add_argument("--output-dir", default="outputs/reve_set_head_selection_s7")
    set_head.set_defaults(handler=_select_set_head_command)

    closure = subparsers.add_parser(
        "reference-closure",
        help="Audit q4 full-reference sensitivity and native class collapse",
    )
    closure.add_argument("--full-run", required=True)
    closure.add_argument("--native-run", required=True)
    closure.add_argument("--selection", required=True)
    closure.add_argument("--output-dir", default="outputs/reve_set_reference_closure_s7")
    closure.add_argument("--bootstrap-resamples", type=int, default=10000)
    closure.add_argument("--bootstrap-confidence", type=float, default=0.95)
    closure.add_argument("--bootstrap-seed", type=int, default=20260715)
    closure.set_defaults(handler=_reference_closure_command)

    geometry = subparsers.add_parser(
        "reference-geometry",
        help="Audit Cz/Pz/Fz effects within fixed native montages",
    )
    geometry.add_argument("--run", required=True)
    geometry.add_argument("--e7d-full-run", required=True)
    geometry.add_argument("--selection", required=True)
    geometry.add_argument("--output-dir", default="outputs/reve_set_reference_geometry_audit_s7")
    geometry.add_argument("--bootstrap-resamples", type=int, default=10000)
    geometry.add_argument("--bootstrap-confidence", type=float, default=0.95)
    geometry.add_argument("--bootstrap-seed", type=int, default=20260715)
    geometry.set_defaults(handler=_reference_geometry_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

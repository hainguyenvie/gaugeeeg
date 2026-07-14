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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

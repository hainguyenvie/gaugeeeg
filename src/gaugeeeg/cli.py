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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gaugeeeg", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the reference-shift experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--encoder", choices=["bandpower", "reve"])
    run.add_argument("--device", help="auto, cpu, cuda, or a torch device such as cuda:1")
    run.add_argument("--output-dir")
    run.add_argument("--force-recompute", action="store_true")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

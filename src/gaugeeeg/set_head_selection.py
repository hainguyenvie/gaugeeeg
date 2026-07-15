"""Validation-only selection for the E7c variable-set readout gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd


def select_set_head(
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    expected_queries: Sequence[int] = (4, 8, 16),
    clean_gate: float = 0.45,
) -> pd.DataFrame:
    """Select query count by validation BAcc, with smaller count breaking ties."""
    rows = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "metrics.csv"
        metrics = pd.read_csv(path)
        car = metrics.loc[metrics["test_view"].astype(str).str.casefold() == "car"]
        if len(car) != 1:
            raise ValueError(f"{path} must contain exactly one CAR row")
        row = car.iloc[0]
        if str(row["probe"]).casefold() != "reve_set":
            raise ValueError(f"{path} was not produced by probe: reve_set")
        rows.append(
            {
                "run_dir": str(run_dir),
                "set_queries": int(row["set_queries"]),
                "selected_epoch": int(row["selected_epoch"]),
                "validation_balanced_accuracy": float(row["validation_balanced_accuracy"]),
            }
        )

    result = pd.DataFrame(rows).sort_values("set_queries").reset_index(drop=True)
    observed = result["set_queries"].tolist()
    expected = sorted(int(value) for value in expected_queries)
    if observed != expected:
        raise ValueError(f"Expected exactly one run for query counts {expected}, observed {observed}")
    selected = result.sort_values(
        ["validation_balanced_accuracy", "set_queries"],
        ascending=[False, True],
    ).iloc[0]
    result["selected_by_validation"] = result["set_queries"] == int(selected["set_queries"])
    selected_metrics = pd.read_csv(Path(str(selected["run_dir"])) / "metrics.csv")
    selected_car = selected_metrics.loc[
        selected_metrics["test_view"].astype(str).str.casefold() == "car"
    ].iloc[0]
    selected_clean_bacc = float(selected_car["balanced_accuracy"])
    result["clean_test_balanced_accuracy"] = pd.NA
    result.loc[result["selected_by_validation"], "clean_test_balanced_accuracy"] = selected_clean_bacc
    clean_passed = selected_clean_bacc >= clean_gate
    summary = {
        "stage": "E7c variable-set readout gate",
        "selection_data": "validation subjects only",
        "selection_metric": "CAR validation balanced accuracy",
        "tie_break": "smaller query count",
        "expected_query_grid": expected,
        "selected_queries": int(selected["set_queries"]),
        "selected_epoch": int(selected["selected_epoch"]),
        "selected_validation_balanced_accuracy": float(selected["validation_balanced_accuracy"]),
        "selected_clean_test_balanced_accuracy": selected_clean_bacc,
        "clean_gate_threshold": float(clean_gate),
        "clean_gate_passed": bool(clean_passed),
        "decision": "proceed_to_native_montage" if clean_passed else "stop_readout_still_too_weak",
        "leakage_control": (
            "Native montage rows are not generated until after query count is frozen; "
            "clean test BAcc is used only as a feasibility gate."
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination / "set_head_grid.csv", index=False)
    with (destination / "set_head_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return result

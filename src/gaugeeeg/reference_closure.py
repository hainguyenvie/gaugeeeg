"""E7d closure audit for full-reference and native-montage class shifts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _load_views(run_dir: str | Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    path = Path(run_dir) / "predictions.csv"
    frame = pd.read_csv(path)
    required = {"test_view", "trial_index", "subject_id", "y_true", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    probability_columns = [column for column in frame if column.startswith("prob_")]
    views = {
        str(view).casefold(): group.sort_values(["subject_id", "trial_index"]).reset_index(drop=True)
        for view, group in frame.groupby("test_view", sort=False)
    }
    identifiers = ["subject_id", "trial_index", "y_true"]
    first = next(iter(views.values()))
    for view, current in views.items():
        if not np.array_equal(first[identifiers].to_numpy(), current[identifiers].to_numpy()):
            raise RuntimeError(f"Prediction trials for {view} are not aligned in {path}")
    return views, probability_columns


def _recall(frame: pd.DataFrame, class_index: int) -> float:
    subset = frame[frame["y_true"] == class_index]
    return float(np.mean(subset["y_pred"].to_numpy() == class_index))


def _class_auroc(frame: pd.DataFrame, probability_column: str, class_index: int) -> float:
    return float(roc_auc_score(frame["y_true"] == class_index, frame[probability_column]))


def _bootstrap_recall_gap(
    base: pd.DataFrame,
    target: pd.DataFrame,
    class_index: int,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    y_true = base["y_true"].to_numpy(dtype=np.int64)
    base_prediction = base["y_pred"].to_numpy(dtype=np.int64)
    target_prediction = target["y_pred"].to_numpy(dtype=np.int64)
    subjects = base["subject_id"].to_numpy(dtype=np.int64)
    unique_subjects = np.unique(subjects)
    subject_indices = {subject: np.flatnonzero(subjects == subject) for subject in unique_subjects}
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled = rng.choice(unique_subjects, size=unique_subjects.size, replace=True)
        indices = np.concatenate([subject_indices[subject] for subject in sampled])
        mask = y_true[indices] == class_index
        gaps[index] = np.mean(base_prediction[indices][mask] == class_index) - np.mean(
            target_prediction[indices][mask] == class_index
        )
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_subjects": int(unique_subjects.size),
        "n_resamples": int(n_resamples),
        "ci_lower": float(np.quantile(gaps, alpha)),
        "ci_upper": float(np.quantile(gaps, 1.0 - alpha)),
        "probability_gap_positive": float(np.mean(gaps > 0.0)),
    }


def _distribution_diagnostics(frame: pd.DataFrame, n_classes: int) -> dict[str, float | int | bool]:
    counts = np.bincount(frame["y_pred"].to_numpy(dtype=np.int64), minlength=n_classes)
    probability = counts / counts.sum()
    positive = probability[probability > 0.0]
    entropy = float(-(positive * np.log(positive)).sum())
    recalls = [_recall(frame, class_index) for class_index in range(n_classes)]
    top_two_share = float(np.sort(probability)[-2:].sum())
    low_recall_classes = int(np.sum(np.asarray(recalls) < 0.05))
    return {
        "n_predicted_classes": int(np.sum(counts > 0)),
        "prediction_entropy_normalized": entropy / np.log(n_classes),
        "effective_predicted_classes": float(np.exp(entropy)),
        "top_two_predicted_class_share": top_two_share,
        "n_classes_with_recall_below_0_05": low_recall_classes,
        "functional_two_class_collapse": bool(top_two_share >= 0.98 and low_recall_classes >= 2),
    }


def analyze_reference_closure(
    full_run_dir: str | Path,
    native_run_dir: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20260715,
) -> pd.DataFrame:
    """Close the q4 full-reference caveat and expose class-conditional montage failure."""
    full_views, full_probability_columns = _load_views(full_run_dir)
    native_views, native_probability_columns = _load_views(native_run_dir)
    required_full = {"car", "cz", "pz", "fcz"}
    required_native = {"car", "native16@car", "native16@cz"}
    if missing := sorted(required_full - set(full_views)):
        raise ValueError(f"Full-reference run is missing views: {missing}")
    if missing := sorted(required_native - set(native_views)):
        raise ValueError(f"Native run is missing views: {missing}")
    if full_probability_columns != native_probability_columns or not full_probability_columns:
        raise ValueError("Full and native runs require the same complete probability columns")

    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    selected_queries = int(selection["selected_queries"])
    full_metrics = pd.read_csv(Path(full_run_dir) / "metrics.csv")
    native_metrics = pd.read_csv(Path(native_run_dir) / "metrics.csv")
    if set(full_metrics["set_queries"].astype(int)) != {selected_queries}:
        raise ValueError("Full-reference run does not use the validation-selected query count")
    if set(native_metrics["set_queries"].astype(int)) != {selected_queries}:
        raise ValueError("Native run does not use the validation-selected query count")

    identifiers = ["subject_id", "trial_index", "y_true"]
    full_car = full_views["car"]
    native_car = native_views["car"]
    car_trials_match = bool(
        np.array_equal(full_car[identifiers].to_numpy(), native_car[identifiers].to_numpy())
    )
    car_predictions_match = bool(
        car_trials_match
        and np.array_equal(full_car["y_pred"].to_numpy(), native_car["y_pred"].to_numpy())
    )
    if not car_predictions_match:
        raise RuntimeError("Deterministic q4 CAR predictions were not reproduced across E7c/E7d")

    class_names = [column.removeprefix("prob_") for column in full_probability_columns]
    n_classes = len(class_names)
    pairs = [
        ("full_reference", "car", view, full_views["car"], full_views[view])
        for view in ("cz", "pz", "fcz")
    ]
    pairs.extend(
        [
            (
                "montage_effect",
                "car",
                "native16@car",
                native_views["car"],
                native_views["native16@car"],
            ),
            (
                "within_montage_reference",
                "native16@car",
                "native16@cz",
                native_views["native16@car"],
                native_views["native16@cz"],
            ),
        ]
    )

    pair_rows: list[dict[str, float | int | str]] = []
    class_rows: list[dict[str, float | int | str]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    for pair_index, (scope, base_view, target_view, base, target) in enumerate(pairs):
        base_prediction = base["y_pred"].to_numpy(dtype=np.int64)
        target_prediction = target["y_pred"].to_numpy(dtype=np.int64)
        y_true = base["y_true"].to_numpy(dtype=np.int64)
        pair_rows.append(
            {
                "scope": scope,
                "base_view": base_view,
                "target_view": target_view,
                "base_balanced_accuracy": float(balanced_accuracy_score(y_true, base_prediction)),
                "target_balanced_accuracy": float(balanced_accuracy_score(y_true, target_prediction)),
                "balanced_accuracy_gap": float(
                    balanced_accuracy_score(y_true, base_prediction)
                    - balanced_accuracy_score(y_true, target_prediction)
                ),
                "prediction_disagreement": float(np.mean(base_prediction != target_prediction)),
            }
        )
        for class_index, (class_name, probability_column) in enumerate(
            zip(class_names, full_probability_columns, strict=True)
        ):
            base_recall = _recall(base, class_index)
            target_recall = _recall(target, class_index)
            base_auroc = _class_auroc(base, probability_column, class_index)
            target_auroc = _class_auroc(target, probability_column, class_index)
            class_rows.append(
                {
                    "scope": scope,
                    "base_view": base_view,
                    "target_view": target_view,
                    "class_index": class_index,
                    "class_name": class_name,
                    "base_recall": base_recall,
                    "target_recall": target_recall,
                    "recall_gap": base_recall - target_recall,
                    "base_predicted_fraction": float(np.mean(base_prediction == class_index)),
                    "target_predicted_fraction": float(np.mean(target_prediction == class_index)),
                    "base_auroc": base_auroc,
                    "target_auroc": target_auroc,
                    "auroc_gap": base_auroc - target_auroc,
                    "target_mean_true_class_probability": float(
                        target.loc[target["y_true"] == class_index, probability_column].mean()
                    ),
                }
            )
            bootstrap_rows.append(
                {
                    "scope": scope,
                    "base_view": base_view,
                    "target_view": target_view,
                    "class_index": class_index,
                    "class_name": class_name,
                    "point_recall_gap": base_recall - target_recall,
                    **_bootstrap_recall_gap(
                        base,
                        target,
                        class_index,
                        n_resamples=n_resamples,
                        confidence=confidence,
                        seed=bootstrap_seed + pair_index * 100 + class_index,
                    ),
                }
            )

    pairs_frame = pd.DataFrame(pair_rows)
    classes_frame = pd.DataFrame(class_rows)
    full_pairs = pairs_frame[pairs_frame["scope"] == "full_reference"]
    largest_full_class = classes_frame[classes_frame["scope"] == "full_reference"].iloc[
        classes_frame[classes_frame["scope"] == "full_reference"]["recall_gap"].abs().argmax()
    ]
    native16_car = native_views["native16@car"]
    native16_cz = native_views["native16@cz"]
    native_distribution = _distribution_diagnostics(native16_car, n_classes)
    collapsed_classes = [
        class_index for class_index in range(n_classes) if _recall(native16_car, class_index) < 0.05
    ]
    collapsed_class_ranking_preserved = bool(
        collapsed_classes
        and all(
            _class_auroc(native16_car, full_probability_columns[index], index) >= 0.55
            for index in collapsed_classes
        )
    )
    full_max_gap_row = full_pairs.iloc[full_pairs["balanced_accuracy_gap"].abs().argmax()]
    full_max_disagreement_row = full_pairs.iloc[full_pairs["prediction_disagreement"].argmax()]
    summary = {
        "stage": "E7d q4 full-reference closure and class-conditional montage audit",
        "selected_queries_frozen": selected_queries,
        "selection_source": str(selection_path),
        "car_trials_reproduced": car_trials_match,
        "car_predictions_reproduced_exactly": car_predictions_match,
        "full_reference_max_absolute_bacc_gap": float(abs(full_max_gap_row["balanced_accuracy_gap"])),
        "full_reference_max_absolute_bacc_gap_view": str(full_max_gap_row["target_view"]),
        "full_reference_max_prediction_disagreement": float(
            full_max_disagreement_row["prediction_disagreement"]
        ),
        "full_reference_max_prediction_disagreement_view": str(
            full_max_disagreement_row["target_view"]
        ),
        "full_reference_largest_absolute_class_recall_gap": float(
            abs(largest_full_class["recall_gap"])
        ),
        "full_reference_largest_shift_view": str(largest_full_class["target_view"]),
        "full_reference_largest_shift_class": str(largest_full_class["class_name"]),
        "native16_car_balanced_accuracy": float(
            balanced_accuracy_score(native16_car["y_true"], native16_car["y_pred"])
        ),
        "native16_cz_balanced_accuracy": float(
            balanced_accuracy_score(native16_cz["y_true"], native16_cz["y_pred"])
        ),
        "native16_car_vs_cz_prediction_disagreement": float(
            np.mean(native16_car["y_pred"].to_numpy() != native16_cz["y_pred"].to_numpy())
        ),
        "native16_class_names": class_names,
        "native16_car_class_recalls": [
            _recall(native16_car, class_index) for class_index in range(n_classes)
        ],
        "native16_cz_class_recalls": [
            _recall(native16_cz, class_index) for class_index in range(n_classes)
        ],
        "native16_car_per_class_auroc": [
            _class_auroc(native16_car, column, index)
            for index, column in enumerate(full_probability_columns)
        ],
        **native_distribution,
        "collapsed_class_indices": collapsed_classes,
        "collapsed_class_ranking_signal_preserved": collapsed_class_ranking_preserved,
        "audit_status": (
            "closure_complete_functional_two_class_collapse_with_recoverable_ranking_signal"
            if native_distribution["functional_two_class_collapse"]
            and collapsed_class_ranking_preserved
            else "closure_complete"
        ),
        "interpretation_rule": (
            "E7c gates remain unchanged. Functional two-class collapse is a post-hoc diagnostic and "
            "a candidate method target, not a retrospective benchmark rejection rule."
        ),
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pairs_frame.to_csv(destination / "paired_view_metrics.csv", index=False)
    classes_frame.to_csv(destination / "class_conditional_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(destination / "class_recall_bootstrap.csv", index=False)
    with (destination / "reference_closure_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return pairs_frame

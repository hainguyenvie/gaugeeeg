"""Class-conditional audit for paired reference-shift predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _bootstrap_recall_gap(
    y_true: np.ndarray,
    car_prediction: np.ndarray,
    shifted_prediction: np.ndarray,
    subjects: np.ndarray,
    class_index: int,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int | bool]:
    unique_subjects = np.unique(subjects)
    subject_indices = {subject: np.flatnonzero(subjects == subject) for subject in unique_subjects}
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_resamples, dtype=np.float64)
    for bootstrap_index in range(n_resamples):
        sampled_subjects = rng.choice(unique_subjects, size=unique_subjects.size, replace=True)
        indices = np.concatenate([subject_indices[subject] for subject in sampled_subjects])
        class_mask = y_true[indices] == class_index
        car_recall = np.mean(car_prediction[indices][class_mask] == class_index)
        shifted_recall = np.mean(shifted_prediction[indices][class_mask] == class_index)
        gaps[bootstrap_index] = car_recall - shifted_recall

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(gaps, alpha))
    upper = float(np.quantile(gaps, 1.0 - alpha))
    return {
        "n_subjects": int(unique_subjects.size),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "bootstrap_mean_recall_gap": float(gaps.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "probability_gap_positive": float(np.mean(gaps > 0.0)),
        "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
    }


def analyze_class_bias(
    run_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260713,
) -> pd.DataFrame:
    """Audit class recall, prediction prevalence, probability, and confusion shifts."""

    metric_rows: list[dict[str, float | int | str]] = []
    bootstrap_rows: list[dict[str, float | int | str | bool]] = []
    confusion_rows: list[dict[str, float | int | str]] = []

    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        prediction_path = run_dir / "predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        frame = pd.read_csv(prediction_path)
        required = {"probe_seed", "test_view", "trial_index", "subject_id", "y_true", "y_pred"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{prediction_path} is missing columns: {missing}")

        probability_columns = [column for column in frame.columns if column.startswith("prob_")]
        class_names = [column.removeprefix("prob_") for column in probability_columns]
        n_classes = int(frame["y_true"].max()) + 1
        if probability_columns and len(probability_columns) != n_classes:
            raise ValueError(f"{prediction_path} has incomplete probability columns")
        if not class_names:
            class_names = [f"class_{index}" for index in range(n_classes)]

        for probe_seed, seed_frame in frame.groupby("probe_seed"):
            views = [str(view) for view in seed_frame["test_view"].unique()]
            if "car" not in [view.casefold() for view in views]:
                raise ValueError(f"{prediction_path} does not contain a CAR view")
            car = seed_frame[seed_frame["test_view"].str.casefold() == "car"].sort_values("trial_index")
            identifiers = car[["trial_index", "subject_id", "y_true"]].to_numpy()

            for view in views:
                if view.casefold() == "car":
                    continue
                shifted = seed_frame[seed_frame["test_view"] == view].sort_values("trial_index")
                shifted_identifiers = shifted[["trial_index", "subject_id", "y_true"]].to_numpy()
                if not np.array_equal(identifiers, shifted_identifiers):
                    raise RuntimeError(f"Trial alignment differs between CAR and {view} in {prediction_path}")

                y_true = car["y_true"].to_numpy(dtype=np.int64)
                subjects = car["subject_id"].to_numpy(dtype=np.int64)
                car_prediction = car["y_pred"].to_numpy(dtype=np.int64)
                shifted_prediction = shifted["y_pred"].to_numpy(dtype=np.int64)
                car_confusion = confusion_matrix(
                    y_true, car_prediction, labels=np.arange(n_classes), normalize="true"
                )
                shifted_confusion = confusion_matrix(
                    y_true, shifted_prediction, labels=np.arange(n_classes), normalize="true"
                )

                for true_class in range(n_classes):
                    for predicted_class in range(n_classes):
                        confusion_rows.append(
                            {
                                "probe_seed": int(probe_seed),
                                "test_view": view,
                                "true_class": class_names[true_class],
                                "predicted_class": class_names[predicted_class],
                                "car_fraction": float(car_confusion[true_class, predicted_class]),
                                "shifted_fraction": float(shifted_confusion[true_class, predicted_class]),
                                "fraction_change": float(
                                    shifted_confusion[true_class, predicted_class]
                                    - car_confusion[true_class, predicted_class]
                                ),
                            }
                        )

                for class_index, class_name in enumerate(class_names):
                    class_mask = y_true == class_index
                    car_recall = float(np.mean(car_prediction[class_mask] == class_index))
                    shifted_recall = float(np.mean(shifted_prediction[class_mask] == class_index))
                    car_prevalence = float(np.mean(car_prediction == class_index))
                    shifted_prevalence = float(np.mean(shifted_prediction == class_index))
                    probability_change = float("nan")
                    if probability_columns:
                        probability_column = probability_columns[class_index]
                        probability_change = float(
                            shifted[probability_column].mean() - car[probability_column].mean()
                        )
                    metric_rows.append(
                        {
                            "probe_seed": int(probe_seed),
                            "test_view": view,
                            "class_index": class_index,
                            "class_name": class_name,
                            "car_recall": car_recall,
                            "shifted_recall": shifted_recall,
                            "recall_gap": car_recall - shifted_recall,
                            "car_predicted_fraction": car_prevalence,
                            "shifted_predicted_fraction": shifted_prevalence,
                            "predicted_fraction_change": shifted_prevalence - car_prevalence,
                            "mean_probability_change": probability_change,
                            "run_dir": str(run_dir),
                        }
                    )
                    bootstrap_rows.append(
                        {
                            "probe_seed": int(probe_seed),
                            "test_view": view,
                            "class_index": class_index,
                            "class_name": class_name,
                            "point_recall_gap": car_recall - shifted_recall,
                            **_bootstrap_recall_gap(
                                y_true,
                                car_prediction,
                                shifted_prediction,
                                subjects,
                                class_index,
                                n_resamples=n_resamples,
                                confidence=confidence,
                                seed=seed,
                            ),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    aggregate = (
        metrics.groupby(["test_view", "class_index", "class_name"], as_index=False)
        .agg(
            n_probe_seeds=("probe_seed", "nunique"),
            recall_gap_mean=("recall_gap", "mean"),
            recall_gap_std=("recall_gap", "std"),
            predicted_fraction_change_mean=("predicted_fraction_change", "mean"),
            predicted_fraction_change_std=("predicted_fraction_change", "std"),
            mean_probability_change_mean=("mean_probability_change", "mean"),
            mean_probability_change_std=("mean_probability_change", "std"),
        )
        .sort_values(["test_view", "class_index"])
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path / "class_metrics_by_seed.csv", index=False)
    aggregate.to_csv(output_path / "aggregate_class_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output_path / "class_recall_bootstrap.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_path / "confusion_shift.csv", index=False)

    largest = aggregate.loc[aggregate["recall_gap_mean"].abs().idxmax()]
    summary = {
        "probe_seeds": sorted(int(value) for value in metrics["probe_seed"].unique()),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "largest_absolute_mean_recall_shift_view": str(largest["test_view"]),
        "largest_absolute_mean_recall_shift_class": str(largest["class_name"]),
        "largest_absolute_mean_recall_gap": float(largest["recall_gap_mean"]),
        "selection_note": "exploratory maximum; Cz/left_fist is predeclared for the next method screen",
    }
    with (output_path / "class_bias_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return aggregate

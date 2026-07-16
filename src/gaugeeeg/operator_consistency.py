"""E15 training-time observation-operator consistency screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


CAR_ONLY = "car_only"
MULTI_VIEW = "multi_view_ce"
OPERATOR = "operator_consistency"
METHODS = (CAR_ONLY, MULTI_VIEW, OPERATOR)
TRAINING_VIEWS = ("car", "native32@car", "native16@car")
AUDIT_VIEWS = (
    "car",
    "native32@car",
    "native16@car",
    "native32@cz",
    "native16@cz",
    "native32@pz",
    "native16@pz",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_e14(path: str | Path) -> dict[str, Any]:
    summary = _load_json(Path(path))
    if summary.get("stage") != "E14 untouched-probe-seed mean-method confirmation":
        raise ValueError("E15 requires the terminal E14 confirmation summary")
    if summary.get("mean_method_new_seed_confirmation_supported") is not False:
        raise ValueError("E15 is licensed only after the E14 mean-method gate fails")
    if summary.get("physionet_test_subjects_used") is not False:
        raise ValueError("E14 must leave the PhysioNet test subjects untouched")
    if summary.get("next_method_recommendation") != (
        "do_not_tune_on_audit_subjects_revisit_source_only_method"
    ):
        raise ValueError("E14 recommendation changed before the E15 pivot")
    return summary


def _prediction_views(run_dir: Path) -> dict[str, pd.DataFrame]:
    path = run_dir / "validation_predictions.csv"
    frame = pd.read_csv(path)
    required = {
        "split",
        "test_view",
        "trial_index",
        "subject_id",
        "y_true",
        "y_pred",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if set(frame["split"].astype(str).str.casefold()) != {"audit"}:
        raise ValueError(f"{path} must contain audit predictions only")
    views = {
        str(view).casefold(): group.sort_values(
            ["subject_id", "trial_index"]
        ).reset_index(drop=True)
        for view, group in frame.groupby("test_view", sort=False)
    }
    if set(views) != set(AUDIT_VIEWS):
        missing_views = sorted(set(AUDIT_VIEWS) - set(views))
        extra_views = sorted(set(views) - set(AUDIT_VIEWS))
        raise ValueError(
            f"{path} changed the frozen E15 audit grid; "
            f"missing={missing_views}, extra={extra_views}"
        )
    identifiers = ["subject_id", "trial_index", "y_true"]
    anchor = views["car"][identifiers].to_numpy()
    for view, current in views.items():
        if current.duplicated(["subject_id", "trial_index"]).any():
            raise ValueError(f"Duplicate E15 prediction trial for {view} in {path}")
        if not np.array_equal(anchor, current[identifiers].to_numpy()):
            raise RuntimeError(f"Prediction trials are not aligned for {view} in {path}")
        labels = set(current["y_true"].astype(int))
        predictions = set(current["y_pred"].astype(int))
        if labels != {0, 1, 2, 3} or not predictions <= {0, 1, 2, 3}:
            raise ValueError(f"Invalid four-class labels or predictions for {view} in {path}")
    return views


def _validate_run(
    run_dir: str | Path,
    *,
    method: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    root = Path(run_dir)
    summary = _load_json(root / "summary.json")
    expected_training = ["car"] if method == CAR_ONLY else list(TRAINING_VIEWS)
    checks = {
        "encoder": summary.get("encoder") == "reve",
        "probe": summary.get("probe") == "reve_set",
        "q4": summary.get("set_queries") == 4,
        "objective": summary.get("probe_objective") == method,
        "training_views": [
            str(value).casefold() for value in summary.get("training_views", [])
        ]
        == expected_training,
        "audit_views": [
            str(value).casefold()
            for value in summary.get("validation_prediction_views", [])
        ]
        == list(AUDIT_VIEWS),
        "strict_determinism": summary.get("strict_determinism") is True,
        "reference_seed": summary.get("reference_seed") == 7,
        "audit_only": summary.get("validation_predictions_only") is True,
        "prediction_split": summary.get("prediction_split") == "audit",
        "split_disjointness": summary.get("all_subject_splits_pairwise_disjoint")
        is True,
        "test_untouched": summary.get(
            "physionet_test_subjects_used_for_fitting_or_scoring"
        )
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid {method} E15 run {root}: {failed}")
    expected_splits = {
        "train_subjects": list(range(1, 61)),
        "probe_validation_subjects": list(range(61, 71)),
        "audit_subjects": list(range(71, 90)),
        "reserved_test_subjects": list(range(90, 110)),
    }
    changed = [
        name
        for name, expected in expected_splits.items()
        if summary.get(name) != expected
    ]
    if changed:
        raise ValueError(f"Frozen E15 subject splits changed in {root}: {changed}")
    if method == OPERATOR:
        if not np.isclose(float(summary.get("consistency_weight", -1.0)), 1.0):
            raise ValueError("E15 freezes operator consistency_weight=1")
        if summary.get("consistency_view_weights") != [0.0, 0.5, 1.0]:
            raise ValueError("E15 freezes operator weights [0, 0.5, 1]")
    views = _prediction_views(root)
    observed_subjects = set(views["car"]["subject_id"].astype(int))
    if observed_subjects != set(range(71, 90)):
        raise ValueError(
            f"{root} predictions must exactly cover audit subjects 71--89"
        )
    return summary, views


def _recalls(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    n_classes = int(max(labels.max(), predictions.max())) + 1
    return recall_score(
        labels,
        predictions,
        labels=np.arange(n_classes),
        average=None,
        zero_division=0,
    )


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    recalls = _recalls(labels, predictions)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "worst_class_recall": float(recalls.min()),
        **{
            f"class_{index}_recall": float(value)
            for index, value in enumerate(recalls)
        },
    }


def _by_view_rows(
    method: str,
    run_dir: str | Path,
    views: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows = []
    for view in AUDIT_VIEWS:
        frame = views[view]
        labels = frame["y_true"].to_numpy(dtype=np.int64)
        predictions = frame["y_pred"].to_numpy(dtype=np.int64)
        rows.append(
            {
                "method": method,
                "run_dir": str(run_dir),
                "test_view": view,
                "n_trials": int(labels.size),
                "n_subjects": int(frame["subject_id"].nunique()),
                **_metrics(labels, predictions),
            }
        )
    return rows


def _metric_value(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric: str,
) -> float:
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(labels, predictions))
    if metric == "worst_class_recall":
        return float(_recalls(labels, predictions).min())
    raise ValueError(f"Unknown paired E15 metric: {metric}")


def _paired_subject_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    metric: str,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    identifiers = ["subject_id", "trial_index", "y_true"]
    if not np.array_equal(
        candidate[identifiers].to_numpy(), baseline[identifiers].to_numpy()
    ):
        raise RuntimeError("Candidate and baseline audit trials are not aligned")
    labels = candidate["y_true"].to_numpy(dtype=np.int64)
    candidate_predictions = candidate["y_pred"].to_numpy(dtype=np.int64)
    baseline_predictions = baseline["y_pred"].to_numpy(dtype=np.int64)
    subjects = candidate["subject_id"].to_numpy(dtype=np.int64)
    unique_subjects = np.unique(subjects)
    subject_indices = {
        subject: np.flatnonzero(subjects == subject) for subject in unique_subjects
    }
    point_candidate = _metric_value(labels, candidate_predictions, metric)
    point_baseline = _metric_value(labels, baseline_predictions, metric)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled_subjects = rng.choice(
            unique_subjects, size=unique_subjects.size, replace=True
        )
        positions = np.concatenate(
            [subject_indices[subject] for subject in sampled_subjects]
        )
        samples[index] = _metric_value(
            labels[positions], candidate_predictions[positions], metric
        ) - _metric_value(
            labels[positions], baseline_predictions[positions], metric
        )
    alpha = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "candidate_value": point_candidate,
        "baseline_value": point_baseline,
        "candidate_minus_baseline": point_candidate - point_baseline,
        "ci_lower": float(np.quantile(samples, alpha)),
        "ci_upper": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "n_subjects": int(unique_subjects.size),
    }


def _selected_pair(
    pairwise: pd.DataFrame,
    *,
    view: str,
    baseline: str,
    metric: str,
) -> pd.Series:
    selected = pairwise.loc[
        pairwise["test_view"].eq(view)
        & pairwise["baseline"].eq(baseline)
        & pairwise["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one E15 comparison for {view}, {baseline}, {metric}"
        )
    return selected.iloc[0]


def analyze_operator_consistency(
    e14_summary: str | Path,
    car_only_dir: str | Path,
    multi_view_dir: str | Path,
    operator_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_resamples: int = 5000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260721,
    minimum_native16_bacc_gain: float = 0.02,
    maximum_clean_bacc_loss: float = 0.01,
    maximum_worst_recall_loss: float = 0.01,
) -> pd.DataFrame:
    """Screen CAR-teacher consistency without touching reserved test subjects."""

    if bootstrap_resamples < 1 or not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("Invalid E15 bootstrap settings")
    frozen_e14 = _validate_e14(e14_summary)
    run_dirs = {
        CAR_ONLY: car_only_dir,
        MULTI_VIEW: multi_view_dir,
        OPERATOR: operator_dir,
    }
    summaries: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, pd.DataFrame]] = {}
    for method, run_dir in run_dirs.items():
        summaries[method], predictions[method] = _validate_run(
            run_dir, method=method
        )
    probe_seeds = {int(summary["probe_seed"]) for summary in summaries.values()}
    if len(probe_seeds) != 1:
        raise ValueError("All E15 development arms must share one probe seed")

    by_view = pd.DataFrame(
        [
            row
            for method in METHODS
            for row in _by_view_rows(method, run_dirs[method], predictions[method])
        ]
    )
    pairwise_rows: list[dict[str, Any]] = []
    comparison_views = ("car", "native32@car", "native16@car")
    for view_index, view in enumerate(comparison_views):
        for baseline_index, baseline in enumerate((CAR_ONLY, MULTI_VIEW)):
            for metric_index, metric in enumerate(
                ("balanced_accuracy", "worst_class_recall")
            ):
                result = _paired_subject_bootstrap(
                    predictions[OPERATOR][view],
                    predictions[baseline][view],
                    metric=metric,
                    n_resamples=bootstrap_resamples,
                    confidence=bootstrap_confidence,
                    seed=(
                        bootstrap_seed
                        + 100 * view_index
                        + 10 * baseline_index
                        + metric_index
                    ),
                )
                pairwise_rows.append(
                    {
                        "test_view": view,
                        "candidate": OPERATOR,
                        "baseline": baseline,
                        **result,
                    }
                )
    pairwise = pd.DataFrame(pairwise_rows)

    clean_vs_car = _selected_pair(
        pairwise,
        view="car",
        baseline=CAR_ONLY,
        metric="balanced_accuracy",
    )
    native16_vs_car = _selected_pair(
        pairwise,
        view="native16@car",
        baseline=CAR_ONLY,
        metric="balanced_accuracy",
    )
    native16_vs_multi = _selected_pair(
        pairwise,
        view="native16@car",
        baseline=MULTI_VIEW,
        metric="balanced_accuracy",
    )
    native16_recall_vs_multi = _selected_pair(
        pairwise,
        view="native16@car",
        baseline=MULTI_VIEW,
        metric="worst_class_recall",
    )
    native32_vs_multi = _selected_pair(
        pairwise,
        view="native32@car",
        baseline=MULTI_VIEW,
        metric="balanced_accuracy",
    )

    clean_preserved = bool(
        clean_vs_car["candidate_minus_baseline"] >= -maximum_clean_bacc_loss
        and clean_vs_car["ci_lower"] >= -maximum_clean_bacc_loss
    )
    native16_recovers = bool(
        native16_vs_car["candidate_minus_baseline"]
        >= minimum_native16_bacc_gain
        and native16_vs_car["ci_lower"] > 0.0
    )
    rule_beats_augmentation = bool(
        native16_vs_multi["candidate_minus_baseline"] > 0.0
        and native16_vs_multi["ci_lower"] > 0.0
        and native16_recall_vs_multi["candidate_minus_baseline"]
        >= -maximum_worst_recall_loss
        and native16_recall_vs_multi["ci_lower"]
        >= -maximum_worst_recall_loss
        and native32_vs_multi["candidate_minus_baseline"] >= 0.0
    )
    development_gate = bool(
        clean_preserved and native16_recovers and rule_beats_augmentation
    )
    multi_native16 = by_view.loc[
        by_view["method"].eq(MULTI_VIEW)
        & by_view["test_view"].eq("native16@car"),
        "balanced_accuracy",
    ].iloc[0]
    car_native16 = by_view.loc[
        by_view["method"].eq(CAR_ONLY)
        & by_view["test_view"].eq("native16@car"),
        "balanced_accuracy",
    ].iloc[0]
    augmentation_helps = bool(multi_native16 > car_native16)
    if development_gate:
        recommendation = "freeze_e15_and_run_reserved_test_multiseed_once"
    elif augmentation_helps:
        recommendation = "retain_multiview_ce_drop_unjustified_rule_loss"
    else:
        recommendation = "stop_current_readout_redesign_encoder_or_data_rule"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_view.to_csv(output / "operator_consistency_by_view.csv", index=False)
    pairwise.to_csv(output / "operator_consistency_pairwise.csv", index=False)
    manifest = pd.DataFrame(
        [
            {
                "method": method,
                "run_dir": str(run_dirs[method]),
                "probe_seed": int(summaries[method]["probe_seed"]),
                "reference_seed": int(summaries[method]["reference_seed"]),
                "selected_epoch": int(summaries[method]["selected_epoch"]),
                "probe_validation_balanced_accuracy": float(
                    summaries[method]["validation_balanced_accuracy"]
                ),
                "test_subjects_used": False,
            }
            for method in METHODS
        ]
    )
    manifest.to_csv(output / "operator_consistency_manifest.csv", index=False)
    summary = {
        "stage": "E15 post-hoc source-only operator-consistency development screen",
        "audit_status": "development_and_falsification_only",
        "e14_mean_method_confirmation_supported": frozen_e14[
            "mean_method_new_seed_confirmation_supported"
        ],
        "probe_seed": probe_seeds.pop(),
        "reference_seed": 7,
        "candidate": OPERATOR,
        "baselines": [CAR_ONLY, MULTI_VIEW],
        "training_views": list(TRAINING_VIEWS),
        "operator_rule": (
            "Full-CAR is a stop-gradient teacher; native32/native16 students "
            "receive KL weights 0.5/1.0 in addition to supervised multi-view CE."
        ),
        "consistency_weight": 1.0,
        "consistency_view_weights": [0.0, 0.5, 1.0],
        "train_subjects": list(range(1, 61)),
        "probe_validation_subjects": list(range(61, 71)),
        "development_audit_subjects": list(range(71, 90)),
        "reserved_test_subjects": list(range(90, 110)),
        "physionet_test_subjects_used": False,
        "minimum_native16_bacc_gain": minimum_native16_bacc_gain,
        "maximum_clean_bacc_loss": maximum_clean_bacc_loss,
        "maximum_worst_recall_loss": maximum_worst_recall_loss,
        "clean_car_preserved": clean_preserved,
        "native16_recovery_vs_car_only_supported": native16_recovers,
        "operator_rule_beats_multi_view_ce": rule_beats_augmentation,
        "multi_view_augmentation_improves_native16_point": augmentation_helps,
        "operator_consistency_development_gate_supported": development_gate,
        "paper_level_claim_supported": False,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_confidence": bootstrap_confidence,
        "decision_rule": (
            "On development audit subjects only: preserve clean CAR within 0.01 "
            "including the paired interval; improve native16 CAR BAcc by at least "
            "0.02 versus CAR-only with an interval above zero; beat multi-view CE "
            "on native16 BAcc with an interval above zero; preserve native16 worst-"
            "class recall within 0.01; and do not lose native32 BAcc by point estimate."
        ),
        "next_method_recommendation": recommendation,
    }
    with (output / "operator_consistency_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    return pairwise

"""Linear-probe and paired-representation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProbeResult:
    model: Pipeline
    selected_c: float
    validation_balanced_accuracy: float


def _make_probe(c: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c),
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def fit_probe(
    train_x: NDArray[np.floating],
    train_y: NDArray[np.integer],
    val_x: NDArray[np.floating],
    val_y: NDArray[np.integer],
    *,
    c_grid: list[float],
    seed: int,
) -> ProbeResult:
    if not c_grid:
        raise ValueError("c_grid must not be empty")
    candidates: list[tuple[float, float]] = []
    for c in sorted(float(value) for value in c_grid):
        candidate = _make_probe(c, seed)
        candidate.fit(train_x, train_y)
        score = balanced_accuracy_score(val_y, candidate.predict(val_x))
        candidates.append((float(score), c))
    best_score, best_c = max(candidates, key=lambda item: (item[0], -item[1]))

    final_model = _make_probe(best_c, seed)
    final_model.fit(np.concatenate([train_x, val_x]), np.concatenate([train_y, val_y]))
    return ProbeResult(final_model, best_c, best_score)


def classification_metrics(
    model: Pipeline,
    x: NDArray[np.floating],
    y: NDArray[np.integer],
) -> dict[str, float]:
    prediction = model.predict(x)
    result = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
    }
    try:
        probability = model.predict_proba(x)
        result["macro_auroc_ovr"] = float(roc_auc_score(y, probability, multi_class="ovr", average="macro"))
    except ValueError:
        result["macro_auroc_ovr"] = float("nan")
    return result


def paired_cosine(x: NDArray[np.floating], y: NDArray[np.floating], eps: float = 1e-12) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape != y_array.shape:
        raise ValueError(f"Paired features must have equal shapes: {x_array.shape} != {y_array.shape}")
    numerator = np.sum(x_array * y_array, axis=1)
    denominator = np.linalg.norm(x_array, axis=1) * np.linalg.norm(y_array, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, eps)))


def relative_l2_drift(x: NDArray[np.floating], y: NDArray[np.floating], eps: float = 1e-12) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape != y_array.shape:
        raise ValueError(f"Paired features must have equal shapes: {x_array.shape} != {y_array.shape}")
    drift = np.linalg.norm(x_array - y_array, axis=1)
    scale = np.linalg.norm(x_array, axis=1)
    return float(np.mean(drift / np.maximum(scale, eps)))


def linear_cka(x: NDArray[np.floating], y: NDArray[np.floating], eps: float = 1e-12) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("CKA inputs must contain the same number of samples")
    x_centered = x_array - x_array.mean(axis=0, keepdims=True)
    y_centered = y_array - y_array.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x_centered.T @ y_centered, ord="fro") ** 2
    self_x = np.linalg.norm(x_centered.T @ x_centered, ord="fro")
    self_y = np.linalg.norm(y_centered.T @ y_centered, ord="fro")
    return float(cross / max(self_x * self_y, eps))


def representation_metrics(clean: NDArray[np.floating], shifted: NDArray[np.floating]) -> dict[str, float]:
    return {
        "paired_cosine_to_car": paired_cosine(clean, shifted),
        "relative_l2_to_car": relative_l2_drift(clean, shifted),
        "linear_cka_to_car": linear_cka(clean, shifted),
    }

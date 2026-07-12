"""Fast algebraic checks that require no EEG download."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .referencing import (
    common_average,
    pairwise_difference,
    reference_matrix,
    single_reference,
    weighted_reference,
)


def run_checks(seed: int = 7) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(16, 8, 256)).astype(np.float32)
    car = common_average(latent)
    shifted = single_reference(car, 3)
    recovered = common_average(shifted)

    weights = rng.dirichlet(np.ones(car.shape[-2]))
    linearly_referenced = weighted_reference(car, weights)
    recovered_linear = common_average(linearly_referenced)

    pair_clean = pairwise_difference(car, 1, 6)
    pair_shifted = pairwise_difference(shifted, 1, 6)
    matrix = reference_matrix(weights)
    annihilation = matrix @ np.ones(matrix.shape[0])

    result: dict[str, float | bool] = {
        "car_recovery_max_abs_error": float(np.max(np.abs(car - recovered))),
        "linear_reference_recovery_max_abs_error": float(np.max(np.abs(car - recovered_linear))),
        "pairwise_invariance_max_abs_error": float(np.max(np.abs(pair_clean - pair_shifted))),
        "reference_matrix_ones_norm": float(np.linalg.norm(annihilation)),
    }
    tolerance = 1e-5
    result["passed"] = all(float(value) < tolerance for value in result.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run_checks(args.seed)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Channel-adaptation operators applied before frozen EEG encoders.

The spherical-spline implementation follows the Perrin interpolation system
used by MNE-Python, while keeping the numerical core dependency-light and
directly testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import eval_legendre

from .referencing import common_average

FloatArray = NDArray[np.floating]


def _normalized_positions(
    channel_names: Sequence[str],
    positions: Mapping[str, ArrayLike],
) -> NDArray[np.float64]:
    lookup = {str(name).casefold(): np.asarray(value, dtype=np.float64) for name, value in positions.items()}
    missing = [str(name) for name in channel_names if str(name).casefold() not in lookup]
    if missing:
        raise ValueError(f"No standard sensor positions for channels: {missing}")
    coordinates = np.stack([lookup[str(name).casefold()] for name in channel_names])
    if coordinates.shape != (len(channel_names), 3):
        raise ValueError("Sensor positions must be three-dimensional vectors")
    norms = np.linalg.norm(coordinates, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Sensor positions must have non-zero length")
    return coordinates / norms


@lru_cache(maxsize=1)
def standard_1005_positions() -> dict[str, NDArray[np.float64]]:
    """Load MNE's standard 10-05 sensor coordinates, keyed case-insensitively."""

    try:
        import mne
    except ImportError as exc:
        raise RuntimeError(
            'Spherical-spline adaptation requires MNE. Install with: pip install -e ".[data]"'
        ) from exc
    montage = mne.channels.make_standard_montage("standard_1005")
    channel_positions = montage.get_positions()["ch_pos"]
    return {
        str(name).casefold(): np.asarray(position, dtype=np.float64)
        for name, position in channel_positions.items()
    }


def _spherical_spline_g(
    cosine_angles: NDArray[np.float64],
    *,
    stiffness: int,
    n_terms: int,
) -> NDArray[np.float64]:
    values = np.zeros_like(cosine_angles, dtype=np.float64)
    for degree in range(1, n_terms + 1):
        coefficient = (2 * degree + 1) / (degree**stiffness * (degree + 1) ** stiffness * 4.0 * np.pi)
        values += coefficient * eval_legendre(degree, cosine_angles)
    return values


def spherical_spline_interpolation_matrix(
    observed_positions: ArrayLike,
    target_positions: ArrayLike,
    *,
    alpha: float = 1e-5,
    stiffness: int = 4,
    n_terms: int = 50,
) -> NDArray[np.float64]:
    """Return weights mapping observed channels to target sensor positions."""

    observed = np.asarray(observed_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 3 or observed.shape[0] < 2:
        raise ValueError("observed_positions must have shape (at least 2, 3)")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target_positions must have shape (n_targets, 3)")
    if alpha < 0.0 or stiffness < 1 or n_terms < 1:
        raise ValueError("Invalid spherical-spline hyperparameters")
    observed_norm = np.linalg.norm(observed, axis=1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=1, keepdims=True)
    if np.any(observed_norm == 0.0) or np.any(target_norm == 0.0):
        raise ValueError("Sensor positions must have non-zero length")
    observed = observed / observed_norm
    target = target / target_norm

    source_cosines = np.clip(observed @ observed.T, -1.0, 1.0)
    target_cosines = np.clip(target @ observed.T, -1.0, 1.0)
    source_g = _spherical_spline_g(source_cosines, stiffness=stiffness, n_terms=n_terms)
    target_g = _spherical_spline_g(target_cosines, stiffness=stiffness, n_terms=n_terms)
    n_observed = observed.shape[0]
    system = np.empty((n_observed + 1, n_observed + 1), dtype=np.float64)
    system[:n_observed, :n_observed] = source_g + alpha * np.eye(n_observed)
    system[:n_observed, n_observed] = 1.0
    system[n_observed, :n_observed] = 1.0
    system[n_observed, n_observed] = 0.0
    target_system = np.concatenate([target_g, np.ones((target.shape[0], 1), dtype=np.float64)], axis=1)
    return target_system @ np.linalg.pinv(system)[:, :n_observed]


def spherical_spline_to_montage(
    x_uv: ArrayLike,
    observed_channel_names: Sequence[str],
    target_channel_names: Sequence[str],
    *,
    positions: Mapping[str, ArrayLike] | None = None,
    alpha: float = 1e-5,
    stiffness: int = 4,
    n_terms: int = 50,
) -> NDArray[np.float32]:
    """Fill missing target channels while retaining measured channels exactly."""

    array = np.asarray(x_uv)
    observed_names = tuple(str(name) for name in observed_channel_names)
    target_names = tuple(str(name) for name in target_channel_names)
    if array.ndim < 2 or array.shape[-2] != len(observed_names):
        raise ValueError("EEG channel axis must match observed_channel_names")
    observed_keys = [name.casefold() for name in observed_names]
    target_keys = [name.casefold() for name in target_names]
    if len(set(observed_keys)) != len(observed_keys) or len(set(target_keys)) != len(target_keys):
        raise ValueError("Channel names must be unique after case normalization")
    if not set(observed_keys).issubset(target_keys):
        extra = [
            name for name, key in zip(observed_names, observed_keys, strict=True) if key not in target_keys
        ]
        raise ValueError(f"Observed channels are absent from target montage: {extra}")
    if observed_keys == target_keys:
        return array.astype(np.float32, copy=False)

    position_bank = standard_1005_positions() if positions is None else positions
    observed_positions = _normalized_positions(observed_names, position_bank)
    missing_indices = [index for index, key in enumerate(target_keys) if key not in set(observed_keys)]
    missing_names = [target_names[index] for index in missing_indices]
    missing_positions = _normalized_positions(missing_names, position_bank)
    weights = spherical_spline_interpolation_matrix(
        observed_positions,
        missing_positions,
        alpha=alpha,
        stiffness=stiffness,
        n_terms=n_terms,
    )

    output = np.empty((*array.shape[:-2], len(target_names), array.shape[-1]), dtype=np.float64)
    observed_lookup = {key: index for index, key in enumerate(observed_keys)}
    for target_index, target_key in enumerate(target_keys):
        if target_key in observed_lookup:
            output[..., target_index, :] = array[..., observed_lookup[target_key], :]
    interpolated = np.einsum("mo,...ot->...mt", weights, array, optimize=True)
    output[..., missing_indices, :] = interpolated
    return output.astype(np.float32)


def adapt_observation_channels(
    x_uv: ArrayLike,
    observed_channel_names: Sequence[str],
    target_channel_names: Sequence[str],
    defense: str,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Apply the configured pre-encoder defense and return aligned names."""

    key = defense.casefold()
    observed_names = tuple(str(name) for name in observed_channel_names)
    if key == "none":
        return np.asarray(x_uv), observed_names
    if key == "car_canonicalize":
        return common_average(x_uv), observed_names
    if key in {"spherical_spline", "spherical_spline_interpolation"}:
        return (
            spherical_spline_to_montage(
                x_uv,
                observed_names,
                target_channel_names,
            ),
            tuple(str(name) for name in target_channel_names),
        )
    raise ValueError(f"Unknown defense: {defense}")


def channel_adaptation_metadata(defense: str) -> dict[str, object]:
    key = defense.casefold()
    if key in {"spherical_spline", "spherical_spline_interpolation"}:
        return {
            "method": "spherical_spline_interpolation",
            "montage": "standard_1005",
            "alpha": 1e-5,
            "stiffness": 4,
            "n_legendre_terms": 50,
            "measured_channels_preserved": True,
        }
    return {"method": key}

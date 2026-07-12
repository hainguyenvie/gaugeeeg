"""Physically valid linear EEG re-referencing operations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


def _as_float_array(x: ArrayLike) -> FloatArray:
    array = np.asarray(x)
    if array.ndim < 2:
        raise ValueError("EEG input must have at least channel and time dimensions")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    return array


def normalize_weights(weights: ArrayLike, n_channels: int, *, atol: float = 1e-8) -> FloatArray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size != n_channels:
        raise ValueError(f"Expected {n_channels} reference weights, got {weights.size}")
    total = float(weights.sum())
    if abs(total) <= atol:
        raise ValueError("Reference weights must have a non-zero sum")
    return weights / total


def reference_matrix(weights: ArrayLike) -> FloatArray:
    """Return R = I - 1 w^T with normalized weights satisfying w^T 1 = 1."""

    raw = np.asarray(weights).reshape(-1)
    w = normalize_weights(raw, raw.size)
    ones = np.ones((w.size, 1), dtype=np.float64)
    return np.eye(w.size, dtype=np.float64) - ones @ w[None, :]


def weighted_reference(x: ArrayLike, weights: ArrayLike) -> FloatArray:
    """Subtract a weighted reference signal along the penultimate (channel) axis."""

    array = _as_float_array(x)
    w = normalize_weights(weights, array.shape[-2]).astype(array.dtype, copy=False)
    reference = np.einsum("...ct,c->...t", array, w)
    return array - reference[..., None, :]


def common_average(x: ArrayLike) -> FloatArray:
    """Project EEG onto the zero-channel-mean (CAR) gauge."""

    array = _as_float_array(x)
    return array - array.mean(axis=-2, keepdims=True)


def single_reference(x: ArrayLike, channel_index: int) -> FloatArray:
    """Reference every channel to one observed electrode."""

    array = _as_float_array(x)
    n_channels = array.shape[-2]
    index = int(channel_index)
    if not -n_channels <= index < n_channels:
        raise IndexError(f"Reference channel index {index} is out of range for {n_channels} channels")
    reference = array[..., index, :]
    return array - reference[..., None, :]


def channel_index(channel_names: Sequence[str], target: str) -> int:
    normalized = {name.casefold(): index for index, name in enumerate(channel_names)}
    key = target.casefold()
    if key not in normalized:
        raise ValueError(f"Reference electrode {target!r} not found. Available: {list(channel_names)}")
    return normalized[key]


def deterministic_linear_weights(channel_names: Sequence[str], seed: int) -> FloatArray:
    """Create a reproducible distributed reference independent of Python hash randomization."""

    material = f"{seed}|{'|'.join(channel_names)}".encode()
    digest = hashlib.sha256(material).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(local_seed)
    return rng.dirichlet(np.ones(len(channel_names), dtype=np.float64))


def apply_reference_view(
    x: ArrayLike,
    channel_names: Sequence[str],
    view: str,
    *,
    seed: int = 0,
) -> FloatArray:
    """Apply a named reference view while preserving channel count and order."""

    view_key = view.strip().casefold()
    if view_key == "car":
        return common_average(x)
    if view_key == "random_linear":
        weights = deterministic_linear_weights(channel_names, seed)
        return weighted_reference(x, weights)
    if view_key.startswith("ref:"):
        electrode = view.split(":", maxsplit=1)[1]
    else:
        electrode = view
    return single_reference(x, channel_index(channel_names, electrode))


def pairwise_difference(x: ArrayLike, first: int, second: int) -> FloatArray:
    array = _as_float_array(x)
    return array[..., first, :] - array[..., second, :]

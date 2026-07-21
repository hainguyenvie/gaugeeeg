"""Gauge-invariant bilateral spectral tokens for motor-imagery EEG."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


# Every pair is present in the locked native16 motor montage.  Midline anchors
# are local to the same anterior/posterior band; two anchors are averaged when
# the sparse montage has no exact FCz location.
BILATERAL_MOTOR_PAIRS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("FC3", "FC4", ("Fz", "Cz")),
    ("FC1", "FC2", ("Fz", "Cz")),
    ("C3", "C4", ("Cz",)),
    ("C1", "C2", ("Cz",)),
    ("CP3", "CP4", ("CPz",)),
    ("CP1", "CP2", ("CPz",)),
)

MOTOR_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (8.0, 13.0),
    (13.0, 20.0),
    (20.0, 30.0),
)

GQBA_MODES = ("none", "spectral_capacity_control", "gqba_odd", "gqba_odd_even")


def _band_log_power(
    signal: FloatArray,
    sfreq: float,
    *,
    bands: Sequence[tuple[float, float]] = MOTOR_BANDS_HZ,
) -> NDArray[np.float32]:
    """Return log periodogram power for fixed motor-rhythm bands."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("EEG signals must include trial and time dimensions")
    if not np.isfinite(values).all() or sfreq <= 0.0:
        raise ValueError("EEG signals and sampling frequency must be finite")
    values = values - values.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(values, axis=-1)
    power = spectrum.real**2 + spectrum.imag**2
    frequencies = np.fft.rfftfreq(values.shape[-1], d=1.0 / float(sfreq))
    features = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            raise ValueError(f"No FFT bins fall inside motor band [{low}, {high}) Hz")
        band_power = power[..., mask].mean(axis=-1)
        features.append(np.log(np.maximum(band_power, np.finfo(np.float64).tiny)))
    return np.stack(features, axis=-1).astype(np.float32)


def bilateral_spectral_tokens(
    x_uv: ArrayLike,
    channel_names: Sequence[str],
    sfreq: float,
    *,
    mode: str,
) -> NDArray[np.float32]:
    """Create six fixed-width bilateral tokens for the locked GQBA screen.

    ``spectral_capacity_control`` exposes left, right, and midline channel
    powers.  It has the same shape and downstream parameter budget as GQBA but
    no reference-invariant spatial rule.  ``gqba_odd`` uses only left-minus-
    right contrasts.  ``gqba_odd_even`` additionally uses bilateral-average-
    minus-midline contrasts, which preserve symmetric motor activity.
    """

    key = str(mode).casefold()
    if key not in GQBA_MODES or key == "none":
        raise ValueError(f"bilateral_spectral_tokens requires a non-none mode from {GQBA_MODES}")
    array = np.asarray(x_uv)
    if array.ndim != 3 or array.shape[1] != len(channel_names):
        raise ValueError("Expected EEG with shape (trials, channels, time)")
    lookup = {str(name).casefold(): index for index, name in enumerate(channel_names)}
    if len(lookup) != len(channel_names):
        raise ValueError("Channel names must be unique after case normalization")
    required = {
        name.casefold()
        for left, right, midlines in BILATERAL_MOTOR_PAIRS
        for name in (left, right, *midlines)
    }
    missing = sorted(required - set(lookup))
    if missing:
        raise ValueError(f"GQBA requires unavailable motor channels: {missing}")

    tokens: list[NDArray[np.float32]] = []
    for left, right, midlines in BILATERAL_MOTOR_PAIRS:
        left_signal = array[:, lookup[left.casefold()], :]
        right_signal = array[:, lookup[right.casefold()], :]
        midline_signal = np.mean(
            np.stack([array[:, lookup[name.casefold()], :] for name in midlines], axis=1),
            axis=1,
        )
        if key == "spectral_capacity_control":
            token = np.concatenate(
                [
                    _band_log_power(left_signal, sfreq),
                    _band_log_power(right_signal, sfreq),
                    _band_log_power(midline_signal, sfreq),
                ],
                axis=-1,
            )
        else:
            # Each spatial row sums to zero, so an arbitrary time-varying
            # common reference cancels exactly before spectral estimation.
            odd = left_signal - right_signal
            odd_power = _band_log_power(odd, sfreq)
            if key == "gqba_odd":
                token = np.concatenate(
                    [odd_power, np.zeros_like(odd_power), np.zeros_like(odd_power)],
                    axis=-1,
                )
            else:
                even = 0.5 * (left_signal + right_signal) - midline_signal
                even_power = _band_log_power(even, sfreq)
                token = np.concatenate([odd_power, even_power, np.zeros_like(odd_power)], axis=-1)
        tokens.append(token)
    return np.stack(tokens, axis=1).astype(np.float32, copy=False)


def gqba_metadata(mode: str) -> dict[str, object]:
    """Return an explicit description of a GQBA auxiliary representation."""

    key = str(mode).casefold()
    if key not in GQBA_MODES:
        raise ValueError(f"Unknown GQBA mode {mode!r}")
    return {
        "mode": key,
        "n_pair_tokens": 0 if key == "none" else len(BILATERAL_MOTOR_PAIRS),
        "token_dimension": 0 if key == "none" else 9,
        "bands_hz": [list(band) for band in MOTOR_BANDS_HZ],
        "pairs": [
            {"left": left, "right": right, "midline": list(midlines)}
            for left, right, midlines in BILATERAL_MOTOR_PAIRS
        ],
        "exact_common_reference_invariance": key in {"gqba_odd", "gqba_odd_even"},
        "contains_antisymmetric_mode": key in {"gqba_odd", "gqba_odd_even"},
        "contains_symmetric_reference_free_mode": key == "gqba_odd_even",
    }

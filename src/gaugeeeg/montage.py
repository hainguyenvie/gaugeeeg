"""Deterministic sparse-montage observation operators.

Two explicit policies are available: legacy sparse views reference the full
recording and then zero-fill unobserved channels, while ``native*`` views first
select the acquired electrodes, reference within that retained montage, and
pass only those signals and coordinates to the encoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .referencing import apply_reference_view

FloatArray = NDArray[np.floating]


# Nested motor-centric subsets. All names occur in the standardized EEGBCI
# 64-channel montage. Case is normalized when matching dataset channels.
SPARSE_MONTAGES: dict[str, tuple[str, ...]] = {
    "sparse8": ("FC3", "FC4", "C3", "C4", "CP3", "CP4", "Cz", "CPz"),
    "sparse16": (
        "FC3", "FC4", "C3", "C4", "CP3", "CP4", "Cz", "CPz",
        "FC1", "FC2", "C1", "C2", "CP1", "CP2", "Fz", "Pz",
    ),
    "sparse32": (
        "FC3", "FC4", "C3", "C4", "CP3", "CP4", "Cz", "CPz",
        "FC1", "FC2", "C1", "C2", "CP1", "CP2", "Fz", "Pz",
        "F3", "F4", "F1", "F2", "FC5", "FC6", "C5", "C6",
        "CP5", "CP6", "P3", "P4", "PO3", "PO4", "O1", "O2",
    ),
}

REGION_DROPS: dict[str, tuple[str, ...]] = {
    "drop_left_motor": ("FC5", "FC3", "FC1", "C5", "C3", "C1", "CP5", "CP3", "CP1"),
    "drop_right_motor": ("FC2", "FC4", "FC6", "C2", "C4", "C6", "CP2", "CP4", "CP6"),
}


@dataclass(frozen=True)
class ObservationView:
    name: str
    reference: str
    montage: str
    channel_policy: str


NATIVE_MONTAGE_ALIASES = {
    "native8": "sparse8",
    "native16": "sparse16",
    "native32": "sparse32",
    "native_drop_left_motor": "drop_left_motor",
    "native_drop_right_motor": "drop_right_motor",
}


def parse_observation_view(view: str) -> ObservationView:
    """Parse ``montage@reference`` or a legacy pure-reference view."""

    name = view.strip()
    if not name:
        raise ValueError("Observation view must not be empty")
    if "@" not in name:
        return ObservationView(name=name, reference=name, montage="full", channel_policy="full")
    montage_key, reference = (part.strip().casefold() for part in name.split("@", maxsplit=1))
    if not montage_key or not reference:
        raise ValueError(f"Invalid observation view {view!r}; expected montage@reference")
    channel_policy = "remove" if montage_key in NATIVE_MONTAGE_ALIASES else "zero"
    montage = NATIVE_MONTAGE_ALIASES.get(montage_key, montage_key)
    if montage not in SPARSE_MONTAGES and montage not in REGION_DROPS:
        allowed = sorted([*SPARSE_MONTAGES, *REGION_DROPS, *NATIVE_MONTAGE_ALIASES])
        raise ValueError(f"Unknown montage {montage!r}; expected one of {allowed}")
    return ObservationView(
        name=name,
        reference=reference,
        montage=montage,
        channel_policy=channel_policy,
    )


def montage_keep_mask(channel_names: Sequence[str], montage: str) -> NDArray[np.bool_]:
    """Return channels observed by a named montage in dataset order."""

    names = [str(name).casefold() for name in channel_names]
    if len(set(names)) != len(names):
        raise ValueError("Channel names must be unique after case normalization")
    key = montage.strip().casefold()
    if key == "full":
        return np.ones(len(names), dtype=bool)
    available = set(names)
    if key in SPARSE_MONTAGES:
        requested = {name.casefold() for name in SPARSE_MONTAGES[key]}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Montage {key} requires unavailable channels: {missing}")
        return np.asarray([name in requested for name in names], dtype=bool)
    if key in REGION_DROPS:
        dropped = {name.casefold() for name in REGION_DROPS[key]}
        missing = sorted(dropped - available)
        if missing:
            raise ValueError(f"Region drop {key} requires unavailable channels: {missing}")
        return np.asarray([name not in dropped for name in names], dtype=bool)
    raise ValueError(f"Unknown montage: {montage}")


def zero_fill_unobserved(
    x: ArrayLike,
    channel_names: Sequence[str],
    montage: str,
) -> FloatArray:
    """Zero unobserved channels while preserving input shape and channel order."""

    array = np.asarray(x)
    if array.ndim < 2 or array.shape[-2] != len(channel_names):
        raise ValueError("EEG channel axis must match channel_names")
    mask = montage_keep_mask(channel_names, montage)
    result = array.copy()
    result[..., ~mask, :] = 0.0
    return result


def apply_observation_view(
    x: ArrayLike,
    channel_names: Sequence[str],
    view: str,
    *,
    seed: int = 0,
) -> FloatArray:
    """Apply the observation policy and its physical reference."""

    observed, _ = prepare_observation_view(x, channel_names, view, seed=seed)
    return observed


def prepare_observation_view(
    x: ArrayLike,
    channel_names: Sequence[str],
    view: str,
    *,
    seed: int = 0,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Return the observed signal and its aligned channel names."""

    specification = parse_observation_view(view)
    names = tuple(str(name) for name in channel_names)
    if specification.montage == "full":
        referenced = apply_reference_view(x, names, specification.reference, seed=seed)
        return referenced, names
    mask = montage_keep_mask(names, specification.montage)
    if specification.channel_policy == "remove":
        observed_names = tuple(name for name, keep in zip(names, mask, strict=True) if keep)
        observed = np.asarray(x)[..., mask, :]
        referenced = apply_reference_view(
            observed,
            observed_names,
            specification.reference,
            seed=seed,
        )
        return referenced, observed_names
    referenced = apply_reference_view(x, names, specification.reference, seed=seed)
    return zero_fill_unobserved(referenced, names, specification.montage), names


def observation_metadata(channel_names: Sequence[str], view: str) -> dict[str, object]:
    specification = parse_observation_view(view)
    mask = montage_keep_mask(channel_names, specification.montage)
    return {
        "reference_view": specification.reference,
        "montage": specification.montage,
        "n_observed_channels": int(mask.sum()),
        "observed_channel_fraction": float(mask.mean()),
        "missing_channel_fill": (
            "none" if mask.all() else "removed" if specification.channel_policy == "remove" else "zero"
        ),
        "channel_policy": specification.channel_policy,
    }

"""Accessible PhysioNetMI download and preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EEGDataset:
    x_uv: NDArray[np.float32]
    y: NDArray[np.int64]
    subjects: NDArray[np.int64]
    channel_names: tuple[str, ...]
    sfreq: float
    label_names: tuple[str, ...]

    def subset(self, subject_ids: list[int]) -> EEGDataset:
        mask = np.isin(self.subjects, np.asarray(subject_ids, dtype=np.int64))
        if not mask.any():
            raise ValueError(f"No trials found for requested subjects: {subject_ids}")
        return EEGDataset(
            x_uv=self.x_uv[mask],
            y=self.y[mask],
            subjects=self.subjects[mask],
            channel_names=self.channel_names,
            sfreq=self.sfreq,
            label_names=self.label_names,
        )


FOUR_CLASS_LABELS = ("left_fist", "right_fist", "both_fists", "both_feet")
UNILATERAL_RUNS = {4, 8, 12}
BILATERAL_RUNS = {6, 10, 14}


def all_subjects(data_config: dict[str, Any]) -> list[int]:
    subjects = (
        set(data_config["train_subjects"])
        | set(data_config["val_subjects"])
        | set(data_config["test_subjects"])
    )
    subjects |= set(data_config.get("audit_subjects", []))
    return sorted(subjects)


def _cache_key(data_config: dict[str, Any]) -> str:
    relevant = {
        "subjects": all_subjects(data_config),
        "runs": data_config["runs"],
        "task": data_config.get("task", "four_class"),
        "resample_hz": data_config.get("resample_hz", 200),
        "epoch_seconds": data_config.get("epoch_seconds", 4.0),
        "highpass_hz": data_config.get("highpass_hz", 0.3),
        "notch_hz": data_config.get("notch_hz", 60.0),
        "max_trials_per_subject": data_config.get("max_trials_per_subject"),
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_cache(path: Path) -> EEGDataset:
    with np.load(path, allow_pickle=False) as cached:
        return EEGDataset(
            x_uv=cached["x_uv"].astype(np.float32, copy=False),
            y=cached["y"].astype(np.int64, copy=False),
            subjects=cached["subjects"].astype(np.int64, copy=False),
            channel_names=tuple(str(name) for name in cached["channel_names"].tolist()),
            sfreq=float(cached["sfreq"].item()),
            label_names=tuple(str(name) for name in cached["label_names"].tolist()),
        )


def _save_cache(path: Path, dataset: EEGDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_uv=dataset.x_uv,
        y=dataset.y,
        subjects=dataset.subjects,
        channel_names=np.asarray(dataset.channel_names),
        sfreq=np.asarray(dataset.sfreq),
        label_names=np.asarray(dataset.label_names),
    )


def _labels_for_run(
    run: int,
    event_codes: NDArray[np.integer],
    t1_code: int,
    t2_code: int,
) -> NDArray[np.int64]:
    labels = np.empty(event_codes.size, dtype=np.int64)
    if run in UNILATERAL_RUNS:
        labels[event_codes == t1_code] = 0
        labels[event_codes == t2_code] = 1
    elif run in BILATERAL_RUNS:
        labels[event_codes == t1_code] = 2
        labels[event_codes == t2_code] = 3
    else:
        raise ValueError(f"Run {run} is not a supported four-class motor-imagery run")
    return labels


def load_physionet_mi(data_config: dict[str, Any], *, force_recompute: bool = False) -> EEGDataset:
    """Download, preprocess, and cache PhysioNetMI in microvolts.

    The clean representation is common-average referenced. Other views are
    generated later from this canonical signal.
    """

    try:
        import mne
        from mne.datasets import eegbci
    except ImportError as exc:
        raise RuntimeError('MNE is required. Install with: pip install -e ".[data]"') from exc

    if data_config.get("task", "four_class") != "four_class":
        raise ValueError("v0.1 implements only task: four_class")

    cache_dir = Path(data_config.get("cache_dir", "data/cache"))
    cache_path = cache_dir / f"physionetmi_{_cache_key(data_config)}.npz"
    if cache_path.exists() and not force_recompute:
        print(f"Loading preprocessed dataset cache: {cache_path}")
        return _load_cache(cache_path)

    root = Path(data_config.get("root", "data/mne"))
    root.mkdir(parents=True, exist_ok=True)
    subjects = all_subjects(data_config)
    runs = [int(run) for run in data_config["runs"]]
    sfreq = float(data_config.get("resample_hz", 200.0))
    epoch_seconds = float(data_config.get("epoch_seconds", 4.0))
    highpass = data_config.get("highpass_hz", 0.3)
    notch = data_config.get("notch_hz", 60.0)
    max_trials = data_config.get("max_trials_per_subject")

    all_x: list[NDArray[np.float32]] = []
    all_y: list[NDArray[np.int64]] = []
    all_groups: list[NDArray[np.int64]] = []
    expected_channels: tuple[str, ...] | None = None

    mne.set_log_level("WARNING")
    for subject in subjects:
        print(f"Preparing PhysioNetMI subject {subject}/{subjects[-1]}")
        paths = eegbci.load_data(subject, runs, path=root, update_path=False, verbose="WARNING")
        if len(paths) != len(runs):
            raise RuntimeError(f"Expected {len(runs)} files for subject {subject}, received {len(paths)}")

        subject_x: list[NDArray[np.float32]] = []
        subject_y: list[NDArray[np.int64]] = []
        for run, raw_path in zip(runs, paths, strict=True):
            raw = mne.io.read_raw_edf(raw_path, preload=True, verbose="ERROR")
            eegbci.standardize(raw)
            raw.pick(picks="eeg")
            raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
            if highpass is not None:
                raw.filter(float(highpass), None, method="iir", verbose="ERROR")
            if notch is not None and float(notch) < raw.info["sfreq"] / 2:
                raw.notch_filter([float(notch)], method="iir", verbose="ERROR")
            if not np.isclose(raw.info["sfreq"], sfreq):
                raw.resample(sfreq, npad="auto", verbose="ERROR")

            events, mapping = mne.events_from_annotations(raw, verbose="ERROR")
            missing = {"T1", "T2"} - set(mapping)
            if missing:
                raise RuntimeError(f"Missing annotations {sorted(missing)} in {raw_path}")
            event_id = {name: mapping[name] for name in ("T1", "T2")}
            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=0.0,
                tmax=epoch_seconds - 1.0 / sfreq,
                baseline=None,
                preload=True,
                reject_by_annotation=True,
                verbose="ERROR",
            )
            x_uv = epochs.get_data(units="uV").astype(np.float32, copy=False)
            event_codes = epochs.events[:, 2]
            y = _labels_for_run(run, event_codes, mapping["T1"], mapping["T2"])

            channels = tuple(epochs.ch_names)
            if expected_channels is None:
                expected_channels = channels
            elif channels != expected_channels:
                raise RuntimeError("Channel order differs across PhysioNetMI files")
            subject_x.append(x_uv)
            subject_y.append(y)

        x_subject = np.concatenate(subject_x, axis=0)
        y_subject = np.concatenate(subject_y, axis=0)
        if max_trials is not None:
            x_subject = x_subject[: int(max_trials)]
            y_subject = y_subject[: int(max_trials)]
        all_x.append(x_subject)
        all_y.append(y_subject)
        all_groups.append(np.full(y_subject.shape, subject, dtype=np.int64))

    if expected_channels is None:
        raise RuntimeError("No EEG data were loaded")
    dataset = EEGDataset(
        x_uv=np.concatenate(all_x, axis=0),
        y=np.concatenate(all_y, axis=0),
        subjects=np.concatenate(all_groups, axis=0),
        channel_names=expected_channels,
        sfreq=sfreq,
        label_names=FOUR_CLASS_LABELS,
    )
    _save_cache(cache_path, dataset)
    print(f"Saved preprocessed dataset cache: {cache_path}")
    return dataset

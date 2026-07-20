# Phase B: frozen-REVE channel adaptation

This stage tests an existing channel-adaptation baseline before introducing a
GaugeEEG method. It does **not** claim spherical-spline interpolation (SSI) as
the project's contribution.

## Why this baseline is next

The Phase-A benchmark selected `joint_multiview_ce` as the strongest eligible
development baseline, but its mean balanced accuracy over
`native16@{CAR,Cz,Pz,Fz}` remains materially below the clean CAR score. A recent
study of channel adaptation for EEG foundation models compares Conv1d
projection, spherical-spline interpolation, source-space decomposition, and
Riemannian recentering, and reports that external adapters can help frozen
flexible-channel models. SSI is the first faithful option for this repository:
it is deterministic, uses only observed signals plus electrode geometry, and
does not require changing or unfreezing REVE.

The implementation follows the Perrin spherical-spline system used in
MNE-Python: standard-1005 unit-sphere positions, stiffness 4, 50 Legendre
terms, and diagonal regularization `alpha=1e-5`. Measured channels are copied
exactly; only absent channels are reconstructed. Interpolation happens after
the native montage has been selected and referenced, so it never reads a
channel that the observation operator declared missing.

Sources:

- Channel Adaptation for EEG Foundation Models: <https://arxiv.org/abs/2604.23091>
- MNE-Python spherical-spline implementation: <https://github.com/mne-tools/mne-python/blob/master/mne/channels/interpolation.py>
- Dynamic Spatial Filtering (requires an end-to-end trainable raw-signal path): <https://arxiv.org/abs/2105.12916>

## Locked matrix

The audit compares four methods, with probe seeds 7, 21, and 42:

| Method | Training | Pre-encoder adapter |
|---|---|---|
| `car_only` | CAR only | none; reused Phase-A control |
| `joint_multiview_ce` | locked joint multiview CE | none; reused Phase-A control |
| `ssi_car_only` | CAR only | spherical-spline interpolation |
| `ssi_joint_multiview_ce` | locked joint multiview CE | spherical-spline interpolation |

All methods are scored on the same 19 audit subjects and the same 12-view
grid. Subjects 90--109 remain unscored here, but they are historically
inspected and therefore are not described as a globally untouched test set.

The primary selection metric remains mean balanced accuracy over
`native16@{CAR,Cz,Pz,Fz}`. The audit now bootstraps this exact mean, paired by
probe seed and subject and paired across the four views. A method is eligible
only if its clean-CAR delta and the lower confidence bound are both at least
`-0.01` versus `car_only`.

## Run

Install the existing data and REVE extras, make sure the Phase-A control
directories are present, then run:

```bash
make channel-adaptation
```

Environment overrides are supported:

```bash
DEVICE=cuda:0 SEEDS="7 21 42" \
CONTROL_ROOT=outputs/reve_benchmark_lock \
OUTPUT_ROOT=outputs/reve_channel_adaptation \
make channel-adaptation
```

The command runs only the six new SSI jobs, reuses the six matching Phase-A
control jobs, and writes the locked aggregate under
`outputs/reve_channel_adaptation/aggregate`.

## External confirmation lock

No PhysioNetMI result from this development stage is paper confirmation. The
external confirmation target is BNCI2014-001: 9 subjects, two sessions, and
four motor-imagery classes (left hand, right hand, feet, tongue). The intended
protocol is subject-wise outer leave-one-subject-out evaluation, with all
method choices and adapter hyperparameters frozen before external labels are
scored. A dataset-specific loader and fold runner should be implemented only
after the Phase-B development decision is frozen; it must not reuse the
PhysioNet class names for the tongue class.

Dataset documentation: <https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html>

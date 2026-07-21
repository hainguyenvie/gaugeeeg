# Gauge-Selective Residual Adapter (GSRA) screen

## Status and hypothesis

This is a locked development-method screen, not external confirmation. The
preceding GQBA experiment increased native16 `both_fists` recall but did not
increase balanced accuracy: the direct residual redistributed predictions
toward the bilateral class and reduced unilateral recall.

GSRA tests one pre-specified correction. A learned scalar gate scales the
reference-invariant GQBA residual for each trial. On non-target training trials
(`left_fist` and `right_fist`), a KL preservation term anchors the final
distribution to the REVE-only distribution. The applied residual is also
constrained to agree across the locked reference/montage views. A binary
rule-applicability loss teaches the gate to activate on the two anatomically
bilateral classes (`both_fists`, `both_feet`) and stay closed on unilateral
movement.

The frozen objective is:

```text
multi-view CE
+ 1.0 * non-target base-to-final KL
+ 0.1 * cross-view applied-residual MSE
+ 0.1 * bilateral rule-applicability gate BCE
```

The gate starts at probability 0.25. These values are fixed before inspecting
subjects 71--89 and must not be tuned on that audit split.

## Matched experimental matrix

All three new arms use the same gated branch and therefore must have identical
trainable and auxiliary parameter counts.

| Method | Auxiliary tokens | Gate | Preservation / residual consistency |
|---|---|---:|---:|
| `gated_spectral_control` | left/right/midline spectral capacity control | yes | 0 / 0 |
| `gqba_gated_ce` | invariant odd+even GQBA | yes | 0 / 0 |
| `gsra` | invariant odd+even GQBA | yes | 1.0 / 0.1 plus 0.1 gate BCE |

The analyzer also reuses the locked `joint_multiview_ce` baseline and the
previous ungated `gqba_odd_even` arm. This separates extra capacity, gated
fusion, invariant features, and the class-preserving objective.

## Run

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make gsra-screen
```

Use `DEVICE=cuda:1` when needed. The script restores the required Phase-A and
GQBA control predictions from their tracked archives, reuses the frozen REVE
feature cache, and launches nine new jobs (three arms by seeds 7/21/42).

The decision artifact is:

```text
outputs/reve_gsra_screen/aggregate/gsra_summary.json
```

Return the three new candidate directories for every seed plus `aggregate`.
The runner also creates `outputs/reve_gsra_screen/validation_predictions.tar.gz`;
commit that archive instead of the nine individual prediction CSV files. The
large existing REVE feature cache is not required.

## Frozen advancement gates

GSRA advances only when all conditions hold:

1. native16 reference-mean BAcc has a paired hierarchical CI above zero versus
   `joint_multiview_ce`;
2. it also beats `gated_spectral_control` and `gqba_gated_ce` with positive
   lower confidence bounds;
3. clean CAR and native32 are noninferior to `joint_multiview_ce` within 0.01
   by point estimate and lower confidence bound;
4. every native16 class-recall point estimate is preserved within 0.01;
5. native16 `both_fists` recall improves by at least 0.03 with a positive
   lower confidence bound; and
6. the validation-only mean gate is larger on the two bilateral target classes
   than on the protected unilateral classes.

A failed gate means retaining `joint_multiview_ce`. Even a complete pass is a
development result and requires one frozen external-dataset confirmation.

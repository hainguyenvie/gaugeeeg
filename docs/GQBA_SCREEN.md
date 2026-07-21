# Gauge-Quotient Bilateral Adapter screen

## Status and hypothesis

GQBA is the first custom representation candidate evaluated after the locked
Phase-A baseline sweep and the Phase-B spherical-spline baseline. It is a
development hypothesis, not a claimed improvement until the frozen analyzer
passes every gate.

The predeclared hypothesis is that an exact reference-invariant bilateral
representation will improve mean balanced accuracy over
`native16@{CAR,Cz,Pz,Fz}`, primarily by recovering `both_fists`, without
reducing clean-CAR or worst-class performance.

## Representation

For every homologous motor pair `(L, R)` and local midline anchor `M`, GQBA
constructs

```text
odd(t)  = L(t) - R(t)
even(t) = (L(t) + R(t))/2 - M(t)
```

Both spatial rows sum to zero. Therefore adding or subtracting any common
time-varying reference signal leaves them unchanged. Odd tokens retain
left-versus-right lateralization; even tokens prevent a pure hemispheric
difference from cancelling bilateral fist/feet activity.

The locked native16-compatible pairs are FC3/FC4, FC1/FC2, C3/C4, C1/C2,
CP3/CP4, and CP1/CP2. Each signal is summarized in 8--13, 13--20, and 20--30
Hz bands. Six spectral tokens enter a small attention branch in parallel with
the frozen REVE q4 set readout. Its final classifier is zero initialized, so
the initial network is exactly the existing REVE head.

This first screen intentionally does not copy NAI-SSL's MRI homologous-region
covariance loss. Equating left and right motor activity could erase the
lateralization needed for left/right fist classification. Cross-sample
covariance remains a later ablation only if the representation screen first
passes.

## Matched controls

The runner compares five methods over probe seeds 7, 21, and 42:

| Method | Role |
|---|---|
| `car_only` | Existing clean/noninferiority control |
| `joint_multiview_ce` | Strongest locked Phase-A baseline |
| `spectral_capacity_control` | Same auxiliary architecture and parameters, but left/right/midline channel power without the quotient rule |
| `gqba_odd` | Exact-invariant antisymmetric representation only |
| `gqba_odd_even` | Proposed complete method |

The analyzer rejects the run if the three auxiliary arms do not have identical
trainable and auxiliary parameter counts.

## Run

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make gqba-screen
```

Use `DEVICE=cuda:1` when needed. Existing frozen REVE caches and Phase-A
controls are reused. The nine new jobs are written under
`outputs/reve_gqba_screen/` and the decision artifact is:

```text
outputs/reve_gqba_screen/aggregate/gqba_summary.json
```

Commit or send back the three candidate directories for every seed plus the
aggregate directory. The required output is much smaller when the existing
REVE feature cache is not included.

## Frozen advancement gates

`gqba_odd_even` advances only when all conditions hold:

1. hierarchical paired CI versus `joint_multiview_ce` is above zero on the
   native16 reference mean;
2. hierarchical paired CI versus `spectral_capacity_control` is above zero;
3. clean CAR is noninferior to `car_only` within 0.01 by point estimate and
   lower confidence bound;
4. native16 both-fists recall improves by at least 0.03; and
5. native16 worst-class recall is preserved within 0.01.

Failure means retaining `joint_multiview_ce`; do not tune the anatomical rule
on subjects 71--89. A passing development result must still be repeated on an
external dataset with all choices frozen.

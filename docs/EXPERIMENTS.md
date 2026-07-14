# Experiment sequence

Run experiments in order. Do not spend GPU time on later stages if an earlier
sanity check fails.

| ID | Purpose | Command | Expected artifact |
|---|---|---|---|
| E0 | Verify reference algebra | `gaugeeeg synthetic` | JSON with errors below `1e-5` |
| E1a | Smoke-test data and metrics | `gaugeeeg run --config configs/smoke.yaml --encoder bandpower` | `outputs/smoke_bandpower/metrics.csv` |
| E1b | Estimate the classical baseline effect | `gaugeeeg run --config configs/pilot.yaml --encoder bandpower` | `outputs/pilot_bandpower/metrics.csv` |
| E2 | Measure frozen REVE sensitivity | `gaugeeeg run --config configs/pilot.yaml --encoder reve --device cuda --output-dir outputs/pilot_reve` | Pilot REVE metrics and drift |
| E3 | Confirm on official split | `gaugeeeg run --config configs/full_physionetmi.yaml --encoder reve --device cuda` | Full-split metrics |
| E4 | Repeat and extend | change seeds; add bipolar/missing-channel views | Confidence intervals and harder benchmark |
| E5 | Audit class-conditional bias | `gaugeeeg class-bias-audit ...` | Stable class-specific recall shifts |
| E6a | Held-out-Cz method screen | seed-7 multi-view CE and rule consistency | Predeclared pilot gate |
| E6b | Multi-seed method confirmation | `make consistency-multiseed` | Hierarchical paired-method bootstrap |
| E6c | Validation-only lambda ablation | `make consistency-lambda-ablation` | Selected λ and held-out-Cz evidence |
| E7a | Reference + sparse-montage screen | `make montage-screen` | Fixed seed-7 limitation screen |
| E7b | Native channel-subset validity screen | `make native-montage-screen` | Corrected benchmark decision |

## E0 acceptance

- CAR recovery maximum absolute error < `1e-5`.
- Pairwise-difference invariance error < `1e-5`.
- Every reference matrix annihilates the all-ones vector.

## E1 acceptance

- E1a completes end-to-end and all four labels occur in every split. Its
  one-subject-per-split score is only a software check and may be below chance.
- In E1b, CAR test performance should be above chance (25%).
- `car_canonicalize` produces nearly identical features for all simple
  reference views.

## E2 interpretation

Focus on the `defense=none` rows. Compare every shifted view with `test_view=car`.
The effect is credible only if task loss and paired representation drift agree.
The `car_canonicalize` rows are a mechanistic positive control, not the final
novel method.

## What to send back

Compress or attach these files:

```text
outputs/pilot_bandpower/metrics.csv
outputs/pilot_bandpower/summary.json
outputs/pilot_bandpower/resolved_config.yaml
outputs/pilot_reve/metrics.csv
outputs/pilot_reve/summary.json
outputs/pilot_reve/resolved_config.yaml
```

Do not send downloaded EDF files or Hugging Face model weights.

## E6b interpretation

E6b holds Cz out of probe training and validation. It trains CAR/Pz/FCz probes
with ordinary multi-view cross-entropy and with the same loss plus
Jensen-Shannon consistency. Seeds 7, 21, and 42 share `reference_seed=7`, the
subject split, frozen REVE features, and all optimizer settings except the
probe initialization/data order seed.

The final comparison resamples both probe seeds and paired test subjects. A
rule-loss contribution is supported only when all consistency runs pass the
30% recall-gap recovery and 1-point clean-loss gates and the hierarchical 95%
CI for `multi_view_ce gap - rule_consistency gap` is above zero. A CI that
crosses zero is explicitly inconclusive even when the point estimate is
positive.

## E6c selection protocol

- Candidate weights: `0, 0.3, 1, 3, 10`; zero denotes multi-view CE.
- Select one global weight using only mean validation BAcc over CAR/Pz/FCz and
  seeds 7/21/42. Break an exact tie toward the smaller weight.
- Cz remains held out from both training and selection.
- After selection, compare the selected weight with lambda zero using the same
  hierarchical seed-and-subject bootstrap as E6b.
- The full test grid is an ablation table, not a second opportunity to choose
  the weight. Do not report a post-hoc best-Cz lambda as the proposed method.

## E7a sparse-montage protocol

- Freeze `lambda=10` from E6c. Montage results cannot change that value.
- Keep training full-montage and unchanged: CAR-only for the clean baseline;
  CAR/Pz/FCz for multi-view CE and rule consistency.
- Apply reference first, then zero-fill missing electrodes while retaining the
  original 64-channel order. Evaluate nested 32/16/8 motor-centric montages and
  asymmetric left/right motor-region drops.
- Fix `sparse16@cz` as the primary view and left-fist as the target class before
  reading results. All other rows diagnose severity and class asymmetry.
- A CAR-only BAcc loss of at least 0.03 on the primary view establishes a useful
  harder stressor. E7a is one-seed exploratory evidence; it cannot support a
  final method claim.
- If the stressor is hard, implement montage-aware training next. Do not spend
  three-seed compute merely reconfirming that the unchanged full-montage method
  fails. If the stressor is weak, revise the observation model first.

## E7b correction after E7a collapse

E7a reached chance because zero-filled channels created an avoidable OOD input
pattern. E7b is a benchmark correction, not a new method result:

- Select retained electrodes, apply the requested reference within that native
  montage, and pass only its signals and coordinates to REVE; never zero-pad.
- Use REVE attention pooling plus a linear probe so native 64/32/16/8-channel
  inputs have one fixed feature dimension.
- Train only on full-montage CAR. Evaluate native 32/16/8 subsets under CAR and
  Cz plus native asymmetric motor-region drops.
- Keep `native16@cz` as the primary target. Decompose its combined loss into a
  montage-only drop and the additional within-montage reference drop.
- Declare the benchmark usable only if clean BAcc is at least 0.45, primary loss
  is at least 0.03, primary BAcc is at least 0.27, at least three classes are
  predicted, normalized prediction entropy is at least 0.30, and no class gets
  more than 95% of predictions.
- Only after this gate passes should a variable-set montage-aware probe or
  adapter be implemented and compared over repeated seeds.

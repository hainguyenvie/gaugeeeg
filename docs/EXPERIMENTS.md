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
| E7c | Variable-set token readout gate | `make set-native-screen` | Clean gate, then corrected benchmark decision |
| E7d | q4 reference/class closure | `make set-reference-closure` | Full-reference and functional-collapse audit |
| E7e | Native reference geometry | `make set-reference-geometry` | Joint-vs-montage-primary scope decision |
| E8 | Validation-only calibration control | `make set-calibration-control` | Readout-vs-representation decision |
| E9 | Reference-bias manifold audit | `make set-bias-manifold` | Unseen-electrode bias predictability |
| E10 | Known-prior/small-batch stress | `make set-prior-stress` | Prior confounding and topology-shrinkage decision |
| E11 | Cross-subject prior identifiability | `make set-prior-identifiability` | Mean-vs-class-uniform robustness decision |
| E12 | Source-only class/operator safeguard | `make set-class-safeguard` | Strict-split class-safety confirmation decision |
| E13 | Post-hoc strongest-baseline audit | `make set-strong-baseline-audit` | Mean-only vs class-uniform advancement decision |

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

E7b did not pass the clean gate. The attention-pooled linear readout reached
0.3988 clean CAR BAcc and 0.2606 on the primary `native16@cz` view. The native
view construction and CAR-canonicalization identities behaved as expected,
but the result cannot determine whether the montage stressor is useful.

## E7c variable-set readout protocol

- Keep the frozen REVE encoder, data split, preprocessing, native subsets,
  primary view, and E7b validity thresholds unchanged.
- Replace released single-query attention pooling with pooling by multihead
  attention (PMA): a learned fixed-size query bank attends to the variable REVE
  token set. Never flatten or zero-pad encoder tokens.
- Predeclare query counts `{4, 8, 16}` and select one using only CAR validation
  BAcc; break an exact tie toward the smaller head.
- Use clean CAR test BAcc only as the predeclared 0.45 feasibility gate. Do not
  generate or inspect native test views until query count is frozen and that
  gate passes.
- If clean passes, retrain the selected deterministic head and evaluate the
  unchanged native suite under no defense and CAR canonicalization.
- E7c validates a readout/benchmark pair. Montage dropout and joint
  gauge/montage consistency are deferred until the native benchmark is usable.

E7c passed the declared benchmark gate at seed 7: q4 validation BAcc was
0.5595, clean CAR was 0.6112, and `native16@cz` was 0.3850. The primary paired
subject gap was 0.2263 with 95% CI [0.1790, 0.2724]. The query grid was close
but not tied: q4 strictly exceeded q8 and q16 on validation.

## E7d reference/class-conditional closure

- Freeze q4 from E7c. Do not repeat query selection or use E7d to modify the
  E7c benchmark gate.
- Train the identical deterministic CAR-only q4 probe and add full-montage
  CAR/Cz/Pz/FCz test views.
- Require exact equality between its CAR predictions and E7c's native-run CAR
  predictions before interpreting any comparison.
- Audit prediction disagreement and class recall/AUROC for three contrasts:
  full-CAR versus each full reference, full-CAR versus native16-CAR, and
  native16-CAR versus native16-Cz.
- Record functional two-class collapse when the two largest predicted classes
  receive at least 98% of predictions and at least two classes have recall
  below 0.05. This is a post-hoc method diagnostic, not retrospective rejection
  of E7c.
- If collapsed classes retain one-vs-rest AUROC above 0.55, treat their decision
  failure as potentially recoverable and proceed to montage-aware learning.

E7d reproduced the E7c q4 CAR predictions exactly. Full Pz produced the
largest full-montage shift: 0.0452 BAcc and 27.1% prediction disagreement.
Native16 CAR lost 0.2268 BAcc relative to full CAR and assigned 99.7% of
predictions to two classes, while the collapsed bilateral classes retained
AUROC 0.615 and 0.681.

## E7e geometry-controlled native-reference protocol

- Keep the q4 architecture, CAR-only training, subject split, probe seed, and
  reference seed fixed. Require exact E7d CAR prediction reproduction.
- Evaluate the complete native32 and native16 Cartesian suite under CAR, Cz,
  Pz, and Fz. Pz/Fz are chosen before observing native results because they are
  retained in both montages and are less central than Cz.
- Report every reference. Do not choose a single E7e winner for method
  evaluation.
- Call joint gauge/montage scope supported if native16 Pz or Fz yields either
  absolute BAcc gap >=0.03 or absolute class-recall gap >=0.10 with a paired
  subject-bootstrap interval excluding zero.
- Prediction disagreement >=0.15 is supporting evidence but cannot alone
  trigger the joint-method decision.
- If the criterion fails, develop montage robustness as the primary method and
  retain gauge consistency only as an auxiliary regularizer.

## E8 validation-only calibration-control protocol

- Freeze the E7c-selected q4 architecture, CAR-only training, splits, seeds,
  preprocessing, and complete E7e native32/native16 reference suite.
- Save raw logits for every validation and test view from the same fitted
  checkpoint. Require validation subjects 71--89 and test subjects 90--109 to
  be disjoint, and require E8 identity predictions to reproduce E7e exactly.
- Fit scalar temperature, zero-sum class bias, and diagonal vector scaling by
  validation NLL only. Class bias is the predeclared primary control because
  E7d/E7e identified relative class-margin shifts; temperature is a negative
  control and vector scaling is a sensitivity analysis. Test labels never
  enter optimization or model choice.
- Report every method under two protocols: target-view-specific calibration
  and leave-one-view-out calibration fitted on the other references in the
  same montage. The first is an oracle known-view baseline; only the second
  tests unseen-reference transfer.
- Temperature cannot change argmax and is therefore an NLL/ECE control. Bias
  and vector scaling are the minimum baselines capable of moving multiclass
  decision boundaries.
- Report BAcc, worst-class recall, macro AUROC, NLL, 15-bin ECE, disagreement,
  per-class recall/frequency/AUROC, fitted parameters, and subject-bootstrap
  BAcc deltas.
- Call calibration explanatory only if it reduces the suite's worst absolute
  within-montage reference recall shift by at least 50% while reducing worst
  native BAcc by no more than 0.01. This gate uses the predeclared class-bias
  method for both protocols; no method or test view is selected after seeing
  test performance.
- If leave-one-view-out passes, prioritize a reference-generalizing calibration
  or gauge-aware readout. If only target-view-specific calibration passes, it
  remains a strong baseline but requires known-view labels. If neither passes,
  proceed to joint gauge/montage representation learning.

E8 passed only the view-specific class-bias gate. It reduced the suite's worst
recall gap from 0.353 to 0.116 and increased worst native BAcc from 0.373 to
0.432. Leave-one-view-out class bias increased the worst recall gap to 0.614,
despite improving BAcc on several views. Pz was the principal over-correction.

## E9 validation-only reference-bias manifold protocol

- Keep q4, preprocessing, seeds, and feature cache fixed. Extract the complete
  native16/native32 electrode-reference grid only for validation subjects;
  the ordinary test path remains full CAR only.
- Split the original validation subjects before analysis: 71--80 fit oracle
  per-reference bias targets and 81--89 evaluate all predictions. These groups
  are disjoint and PhysioNet test subjects 90--109 are not used by the E9
  manifold analysis or its method-selection gate.
- Hold out an electrode identity across both montages. For example, a Pz fold
  excludes native16@Pz and native32@Pz from bias-predictor training.
- Compare identity, global-mean bias, pooled bias (the E8 strategy on the
  expanded grid), nominal 10--20 topology ridge, unlabeled target-batch
  logit-statistics ridge, their concatenation, and a target-label oracle upper
  bound. Ridge alpha is fixed at 1.0.
- Labels from the held-out reference may define its oracle bias for error
  measurement and the upper bound only. A unit test perturbs those labels and
  requires every deployable held-out prediction to remain exactly unchanged.
- Require exact reproduction of every E8 validation view/logit shared with the
  E9 grid before interpreting results.
- The primary combined predictor passes if its leave-one-electrode-out bias
  RMSE is at least 20% lower than the better of global-mean and pooled bias,
  its worst class-recall gap is at least 30% lower than identity, and mean BAcc
  is no more than 0.01 below the better simple baseline. No PhysioNet test
  result selects a method.
- Passing combined/topology conditioning licenses an operator-conditioned
  calibrator. Passing logit-only conditioning licenses label-free batch
  adaptation. If all fail, simple smooth bias-manifold assumptions are not a
  viable method foundation.

E9 passed all three candidate gates. Across 48 held-out target views,
topology/logit/combined ridge achieved bias RMSE 0.223/0.037/0.044 versus
0.559 for global mean. Logit-only and combined raised mean BAcc from 0.420 to
0.462 and reduced the worst recall gap from 0.478 to 0.205/0.181. After
case-normalizing view names, all eight E8 views shared with E9 reproduce
predictions and logits exactly; the original summary reported only the two
CAR views because its matching was case-sensitive.

The E9 oracle bias is not a general label-dependent target. Its three
parameters correlate at absolute 0.996--0.999 with the corresponding centered
logit means. More importantly, the additive-bias NLL objective depends on
labels only through class proportions. Matching the observed empirical prior
reproduces supervised oracle parameters to numerical tolerance; using the
predeclared uniform task prior gives mean bias RMSE 0.0035. Learned logit ridge
is therefore not the baseline to carry forward.

## E10 known-prior and small-batch stress protocol

- Reuse the E9 validation logits; do not run the encoder or use PhysioNet test
  subjects. Keep subjects 71--80 for calibration-batch construction and
  81--89 for task evaluation.
- Use the known four-class uniform task prior. Verify separately that the
  empirical fit prior reproduces the supervised bias optimum, but label this
  as a mathematical diagnostic rather than a deployable input.
- Evaluate random unlabeled batches of 16, 32, 64, 128, 256, 512, and all 900
  fitting trials with 20 fixed resamples except for the deterministic full
  batch.
- At batch sizes 32 and 128, add class-balanced batches. At 128, add four
  dominant-class variants at 40% and 70%. Labels are allowed only to construct
  these controlled stress batches and calculate audit metrics.
- Compare identity, leave-one-electrode-out topology ridge, known-prior
  matching, topology shrinkage, and a supervised oracle. A target electrode is
  excluded across both montages from topology training and shrinkage-weight
  estimation.
- Set the weight on prior matching from source-reference errors only:
  `w_prior = MSE_topology / (MSE_topology + MSE_prior)`. Use random source
  batches to estimate this weight; do not retune it on balanced or skewed
  target conditions.
- The primary condition is random `n=32`. Shrinkage passes only if it reduces
  mean bias RMSE by at least 20% versus prior matching, beats topology-only
  RMSE, reduces mean maximum recall gap by at least 10%, loses no more than
  0.01 mean BAcc, and has a paired reference/resample-bootstrap RMSE-delta
  interval below zero.
- Report prior confounding when severe skew at `n=128` at least doubles RMSE,
  loses 0.03 BAcc, or increases mean maximum recall gap by 0.05 relative to
  balanced batches. This limitation remains mandatory even if shrinkage
  passes.

E10 passed its small-batch gate on the user-reproduced run. At random `n=32`,
topology shrinkage reduced mean bias RMSE by 38.0%, reduced the mean maximum
recall gap by 19.5%, and changed mean BAcc by +0.0022. Its paired RMSE-delta
95% interval was [-0.154, -0.069]. Severe-skew RMSE at `n=128` was 3.00 times
balanced RMSE, confirming the prior-confounding limitation.

## E11 cross-subject prior-identifiability protocol

- Reuse E9 validation logits only. Do not extract features or touch PhysioNet
  test subjects. Keep evaluation subjects 81--89 unchanged.
- Train the soft class-probability model on CAR trials from subjects 71--75.
  Estimate its soft confusion matrix from leave-one-subject-out probabilities.
  Construct adaptation batches only from disjoint subjects 76--80, with source
  and adaptation resampling seeds fixed separately.
- Recover a simplex pseudo-prior by ridge-regularized inversion of the soft
  confusion matrix, anchored to the nominal uniform prior. Compare the fixed
  regularization `1.0` with weak `0.1` and soft-mean ablations; do not select a
  value on held-out target performance.
- Convert the pseudo-prior to an additive target bias, then shrink it toward
  leave-one-electrode-out topology using source-reference errors. Exclude the
  target electrode identity across native16 and native32 throughout fitting.
- Target-adaptation labels may construct random, balanced, 40%, and 70%
  controlled-skew batches and score the audit only. A leakage test perturbs
  held-out target-reference labels and requires all deployable estimates to
  remain exactly unchanged.
- Preserve random `n=32` and balanced `n=128` within 5% bias RMSE, 0.01 BAcc,
  and 0.01 mean maximum-recall-gap tolerances relative to fixed E10 shrinkage.
- Mean severe robustness requires at least 5% RMSE reduction at 70% skew,
  lower RMSE than topology, preserved task metrics, and a paired 95% RMSE-delta
  interval below zero. The strict method gate additionally requires improvement
  for every dominant-class direction. Do not replace this with an average-only
  claim after observing results.

The local reference run supports mean severe robustness (8.95% RMSE reduction;
paired interval [-0.0219, -0.0113]) but not class-uniform robustness. Three of
four dominant-class directions improved; right-fist-dominant RMSE changed from
0.1452 to 0.1482. Therefore E11 is a diagnostic result, and the next method
must add a class-conditional safeguard before a paper-level method claim.

The archived E11 run used subjects 76--80 in both topology fitting and target
adaptation. Treat those numbers as an exploratory failure analysis only. The
current E11 command and E12's nested control fit topology on subjects 71--75,
adapt on 76--80, and evaluate on 81--89.

## E12 source-only class/operator safeguard protocol

- Reuse the E9 validation logits and rerun E11 internally with strict disjoint
  source (71--75), adaptation (76--80), and evaluation (81--89) subjects.
- For each held-out reference identity, construct controlled source examples
  from all other reference identities. Exclude both the outer held reference
  and the current pseudo-target reference from every nested topology fit.
- Fit one diagonal trust cap per class by least squares in the four-class
  zero-sum bias space. Source labels define the oracle training target and
  controlled source conditions; no adaptation/evaluation label fits a cap.
- Apply each cap only as an upper bound on E11's pre-existing scalar trust.
  Thus E12 can fall back toward topology but cannot amplify a pseudo-prior
  update beyond E11.
- Preserve random `n=32` and balanced `n=128` within the same RMSE, BAcc, and
  recall-gap tolerances used by E11. Mean severe robustness also requires at
  least 5% RMSE reduction, lower RMSE than topology-only, and a clustered
  interval below zero.
- Advance to repeated-seed confirmation only if every severe dominant-class
  raw and clustered point direction improves and no two-sided interval detects
  harm. A paper-level class-uniform claim requires every class-specific upper
  interval bound below zero.

The pre-push CPU screen passed the repeated-seed gate: severe RMSE changed from
0.3489 to 0.2013 (42.3% reduction), versus 0.2239 for topology-only. All four
dominant-class point directions improved. The right-fist clustered interval
was approximately [-0.0266, 0.0206], so the stronger paper-level class-uniform
claim remains false pending repeated seeds and an external dataset.

## E13 post-hoc strongest-baseline protocol

- Consume the frozen E12 summary and metric table only. Do not refit E11, E12,
  topology, or any target correction, and do not use target labels for a new
  fitted quantity.
- Compare E12 with fixed E10 shrinkage for continuity, but define strict E11
  operator-confusion shrinkage and topology-only ridge as the two strong
  baselines that must both be addressed.
- Use paired 95% bootstrap intervals over batch repeats and held-out reference
  identities. Broadcast the static topology estimate across repeats rather
  than treating duplicated rows as independent observations.
- Audit random `n=32`, balanced `n=128`, mean severe 70% skew, and every
  severe dominant-class direction. Preserve BAcc and maximum recall gap within
  fixed 0.01 noninferiority margins.
- Eligibility for new independent seeds is deliberately weaker than
  confirmation: require lower point RMSE than both strong baselines, task
  noninferiority intervals, and no detected material harm. Report separately
  whether all RMSE intervals are below zero.
- A class-uniform candidate must also beat both strong baselines for every
  dominant class and have no class-specific harm. Because this rule was
  written after reviewing E12 seed 7, it can reject but cannot confirm a paper
  claim.

The pre-push audit found lower mean E12 RMSE against both strong baselines in
all three regimes. The primary delta versus strict E11 was -0.0076 with 95%
interval [-0.0193, 0.0033], so seed 7 alone does not confirm the mean claim.
Balanced and severe mean intervals were below zero against both baselines.
Right-fist-dominant E12 RMSE was worse than strict E11 by 0.0182, with interval
[0.0006, 0.0361]. E13 therefore advances only the mean-method hypothesis to
new probe seeds and rejects advancement of the current class-uniform claim.

# GaugeEEG Research Conversation Log

This file records the important decisions, evidence, and next actions from the
iterative research process. It is intentionally a concise research log rather
than a verbatim chat transcript. Update it whenever a new experiment changes
the working hypothesis or paper direction.

## 2026-07-12 — Starting point

- The seed paper was *Neuro-Anatomy-Informed Self-Supervised Learning for
  Structural Brain MRI* (NAI-SSL).
- We did not select NAI-SSL as the benchmark baseline because it is not an
  established top-conference anchor. We transferred its core principle:
  scientific rules can define useful transformations and consistency
  constraints for representation learning.
- Candidate directions were: improve a strong existing benchmark, introduce a
  new dataset/rule, or construct a new rule-informed representation-learning
  task.
- Feasibility requirement: public data, available code/weights, clear baselines,
  and a small proof of concept before committing to a paper claim.

## 2026-07-12 — Selected research hypothesis

- Anchor model: REVE, a NeurIPS 2025 EEG foundation model with released code and
  weights.
- Dataset: public PhysioNet EEG Motor Movement/Imagery (PhysioNetMI).
- Proposed rule: EEG voltage is defined only relative to a reference. Physically
  valid re-referencing changes the numerical signal but preserves pairwise
  voltage differences and the underlying physiological event.
- Initial question: does a frozen EEG foundation representation and its
  downstream classifier remain stable when the same recording is expressed
  under another valid reference convention?
- Initial transforms: CAR, Cz, Pz, FCz, and deterministic random linear
  reference.

## 2026-07-13 — First pilot result

- Pilot used 12 subjects and 1,080 trials.
- Algebraic checks passed: CAR canonicalization removed additive reference
  components to numerical precision.
- REVE representation geometry changed under single-electrode references
  (linear CKA fell to approximately 0.72), but the downstream clean classifier
  was at chance (balanced accuracy approximately 0.244).
- Decision: the pilot could not test downstream robustness because the clean
  baseline was not useful. Do not interpret reference accuracy gaps yet.
- Root cause hypothesis: attention-pooled embeddings plus sklearn logistic
  regression did not reproduce REVE's released PhysioNetMI linear-probe setup.

## 2026-07-13 — Official-like clean baseline gate

- Implemented REVE's non-pooled token-level probe shape: pretrained query-token
  initialization, query attention, all frozen tokens, RMSNorm, dropout, and a
  trainable linear head.
- Used the released 70/19/20 subject split and all 9,837 trials.
- Clean CAR result, seed 7:
  - validation balanced accuracy: 0.5132
  - held-out test accuracy: 0.5563
  - held-out test balanced accuracy: 0.5567
  - macro-F1: 0.5521
  - macro-AUROC: 0.8249
  - selected epoch: 9
- The clean gate threshold of 0.45 passed. This result is compatible with the
  scale of REVE's published PhysioNetMI linear-probe results, although it is not
  an exact reproduction because GaugeEEG currently uses AdamW rather than the
  released StableAdamW implementation and has only one seed.
- A warmup edge case was fixed: `warmup_epochs=0` must produce zero warmup steps
  instead of pinning the learning rate near zero.

## 2026-07-13 — Current experiment: reference stress screening

- Train one frozen-REVE token probe on clean CAR using seed 7.
- Evaluate that same trained probe on CAR, Cz, Pz, FCz, and random-linear test
  representations. The probe is not refit for shifted views.
- Reuse clean-gate token caches. Extract only missing shifted test tokens.
- Primary effect: `BAcc(CAR) - BAcc(shifted)`.
- Screening threshold: at least 0.03 absolute balanced-accuracy degradation for
  one valid reference.
- Secondary evidence: macro-F1, AUROC, paired cosine, relative L2, and linear
  CKA.
- If the threshold is met, repeat with seeds 7/21/42 and add statistical
  reporting. If accuracy is stable despite representation drift, test harder
  lawful convention shifts such as bipolar montage, missing electrodes, and
  mixed train/test reference conventions.

## Artifacts expected from the current run

- `outputs/reve_reference_stress/metrics.csv`
- `outputs/reve_reference_stress/summary.json`
- `outputs/reve_reference_stress/resolved_config.yaml`
- `outputs/reve_reference_stress/probe_history_none.csv`
- `outputs/reve_reference_stress/probe_best_none.pt` (local artifact; ignored by
  Git unless explicitly selected)

## 2026-07-13 — Three-seed reference-stress result

- Repeated the CAR-trained token probe with probe seeds 7, 21, and 42.
- Mean clean CAR balanced accuracy was 0.5788.
- Cz was the most consistent fixed stress view: balanced-accuracy gap
  0.0319 +/- 0.0029 using the original population-standard-deviation summary.
- FCz had a 0.0255 mean gap; Pz was less stable; random-linear reference did
  not reduce accuracy because its dense Dirichlet weights produce a reference
  close to CAR.
- Only 2/3 seeds crossed the predeclared 0.03 screening threshold, and the
  per-seed worst view switched between Cz and FCz.
- Representation sensitivity remained clear (single-reference linear CKA
  approximately 0.75--0.77), but downstream degradation was judged
  weak/marginal rather than conclusive.

## 2026-07-13 — Decision and statistical-audit protocol

- Do not use the mean of a separately selected worst view per seed as the
  headline effect. Select one fixed reference across seeds.
- Separate `probe_seed` from `reference_seed`: vary optimization while keeping
  the physical transform and frozen token cache fixed.
- Enable strict PyTorch determinism and repeat seed 7 to audit the earlier
  same-seed clean-score discrepancy.
- Save trial/subject identifiers and predictions, then calculate paired
  subject-cluster bootstrap confidence intervals for BAcc(CAR)-BAcc(shifted).
- Use sample standard deviation (`ddof=1`) in multi-seed aggregation.
- Pure re-referencing remains a positive-control problem because CAR is an
  exact canonicalization when all channels are retained. The likely paper
  pivot is a joint reference-and-montage benchmark with missing channels,
  sparse montages, or bipolar derivations, where CAR alone is insufficient.

## 2026-07-13 — Deterministic subject-level audit result

- The seed-7 run and exact repeat produced byte-identical metrics, predictions,
  and subject metrics. The previous same-seed discrepancy was therefore a CUDA
  determinism issue and is now resolved.
- Fixed-reference Cz BAcc gaps for probe seeds 7/21/42 were 0.0603, 0.0216, and
  0.0238. The mean was 0.0352 with sample standard deviation 0.0217.
- Paired subject-cluster 95% CIs excluded zero for Cz in seeds 7 and 21, but the
  seed-21 lower bound was only 0.0001 and seed 42 included zero. The global
  downstream degradation remains weak/marginal across probe optimization.
- The class-conditional effect was much more stable: under Cz, left-fist recall
  dropped by 0.2069 +/- 0.0130 across seeds, predicted left-fist prevalence
  dropped by 0.1113, and mean left-fist probability dropped by 0.1039.
- Cz simultaneously improved both-fists recall by approximately 0.0956, so
  aggregate BAcc partially cancels the systematic label-geometry shift.
- Reframed limitation: physically equivalent reference conventions induce
  systematic class-conditional decision bias in a frozen EEG foundation model,
  which aggregate performance metrics can hide.

## 2026-07-13 — Held-out-reference consistency screen

- Primary target: Cz-induced left-fist recall gap. Cz must not appear in probe
  training or validation.
- Training views: CAR, Pz, and FCz with aligned trials and fixed
  `reference_seed=7`.
- Baseline A: deterministic CAR-only probe already available.
- Baseline B: supervised multi-view cross-entropy (`multi_view_ce`).
- Proposed method: the same multi-view cross-entropy plus Jensen-Shannon
  consistency across paired reference views (`rule_consistency`).
- Seed-7 success criteria are predeclared: at least 30% relative reduction of
  the held-out-Cz left-fist recall gap and at most 0.01 absolute CAR BAcc loss.
- Attribute novelty to the rule term only if consistency passes and improves
  over multi-view augmentation. If augmentation performs equally well, the
  consistency loss is not a supported contribution.
- Only after the seed-7 gate passes should seeds 21/42 and reference-plus-montage
  experiments be run.

## 2026-07-14 — Seed-7 result and multi-seed confirmation decision

- Both seed-7 methods passed the predeclared gate. Multi-view CE reduced the
  held-out-Cz left-fist recall gap from 0.2217 to 0.1197; rule consistency
  reduced it to 0.0976. CAR BAcc increased rather than decreased.
- Rule consistency reduced the Cz BAcc gap from 0.0603 to 0.0107, compared with
  0.0235 for multi-view CE. Its own subject bootstrap no longer detected a
  nonzero Cz degradation.
- The direct rule-versus-augmentation recall-gap advantage was only 0.0222 in
  seed 7. A post-hoc paired subject bootstrap gave a 95% interval of roughly
  [-0.020, 0.067], so one seed does not support a superiority claim.
- Next action: train only the multi-view CE and rule-consistency probes for
  seeds 21/42, then use a hierarchical bootstrap over probe seeds and paired
  subjects. Do not run a lambda sweep yet.
- Decision rule: call the rule term `supported` only when every consistency run
  passes the recovery/clean gate and the hierarchical 95% interval for direct
  recall-gap recovery is above zero. Otherwise report
  `promising_but_inconclusive` or attribute recovery to augmentation.
- Scope note: the encoder remains frozen. This stage tests a rule-informed
  robust readout, not yet encoder-level representation learning. Harder
  missing-channel/montage experiments follow only after method confirmation.

## 2026-07-14 — Multi-seed consistency result

- Rule consistency passed the predeclared recovery/clean gate in all three
  probe seeds and beat multi-view CE on the held-out-Cz left-fist recall gap in
  all three: improvements were 0.0222, 0.0421, and 0.0111.
- Mean left-fist recall gaps were 0.2069 for CAR-only, 0.0976 for multi-view CE,
  and 0.0724 for rule consistency. Mean Cz BAcc gaps were 0.0352, 0.0107, and
  0.0013, respectively.
- The direct hierarchical recall-gap improvement was 0.0251 with 95% interval
  [-0.0022, 0.0547] and probability of a positive improvement 0.9649. Under the
  predeclared two-sided-CI decision rule, evidence is
  `promising_but_inconclusive`, not `supported`.
- A post-hoc subject audit found positive mean recovery for 12/20 subjects,
  zero for 5/20, and negative recovery for 3/20; the median was approximately
  0.0211. The effect is not explained by one subject, but this audit is
  supporting rather than confirmatory evidence.

## 2026-07-14 — Predeclared lambda-ablation protocol

- Motivation: at selected epochs, the lambda-one consistency term contributes
  only a modest fraction of the total optimization objective; the weak direct
  effect may reflect underweighting rather than a false rule.
- Candidate grid: `lambda = 0, 0.3, 1, 3, 10` over probe seeds 7/21/42. Lambda
  zero is ordinary multi-view CE; existing lambda-zero and lambda-one runs are
  reused, and only nine new probes are trained.
- One global lambda is selected using mean validation BAcc across CAR/Pz/FCz
  and all three seeds. An exact tie selects the smaller lambda. Cz must not be
  read during selection.
- After lambda is frozen, compare it with lambda zero on Cz using the
  hierarchical seed-and-subject bootstrap. The complete Cz grid is reported
  only as an ablation and cannot be used to reselect lambda.
- If validation selects zero, abandon the rule-loss claim. If it selects a
  positive lambda but the held-out comparison remains inconclusive, carry the
  result only as motivation into the harder reference-plus-montage benchmark.

## 2026-07-14 — Validation-selected lambda result

- Validation over CAR/Pz/FCz and seeds 7/21/42 selected lambda 10 without
  reading held-out Cz.
- Against multi-view CE, lambda 10 improved the held-out-Cz left-fist recall
  gap by 0.0310. The hierarchical seed-and-subject 95% interval was
  [0.0052, 0.0618], with probability of positive improvement 0.9888.
- The advantage was positive in all three seeds. Target Cz BAcc increased by
  0.0210 and clean CAR BAcc increased by 0.0185; their hierarchical intervals
  also excluded zero.
- Under the predeclared rule, the rule-loss evidence status is now `supported`.
  Lambda 10 is the best candidate only within the declared grid; it lies at the
  upper boundary and must not be called a global optimum.
- This closes the ideal full-channel reference stage. The encoder is still
  frozen, and CAR remains an exact solution for pure full-channel gauge shifts.

## 2026-07-14 — Predeclared reference-plus-montage screen

- Next limitation: the selected reference-consistency readout has never seen
  missing electrodes, and exact CAR cannot recover signals that were not
  observed.
- E7a keeps lambda 10 and all training choices frozen. It evaluates CAR-only,
  CAR canonicalization, multi-view CE, and rule consistency at seed 7.
- OOD observations are nested motor-centric 32/16/8 channel masks under CAR
  and Cz plus asymmetric left/right motor-region dropout. Reference is applied
  before channels are zero-filled; channel order and tensor shape are retained.
- The primary view is fixed as `sparse16@cz`, with left-fist as target class.
  Other test rows are diagnostic and cannot be used to select another target.
- If CAR-only loses at least 0.03 BAcc, proceed to montage-aware training. This
  one-seed screen is designed to identify a method limitation, not to establish
  a final superiority claim.

## 2026-07-14 — E7a result: zero-fill screen invalidated

- The primary `sparse16@cz` view reduced CAR-only BAcc from 0.5708 to exactly
  0.2500, but every method reached chance on the same view.
- Prediction inspection showed an input-collapse signature: CAR-only predicted
  one class for 99.9% of trials, while multi-view CE and lambda-10 consistency
  predicted that class for 100%.
- Sparse difficulty was non-monotonic, and lambda 10 had zero target-BAcc gain
  over multi-view CE. Its left-fist recall-gap recovery was negative.
- Macro-AUROC remained around 0.57--0.60 despite chance BAcc, indicating that
  ranking information survived while the zero pattern shifted decision bias.
- Decision: do not repeat E7a over more seeds and do not claim a genuine
  missing-channel limitation from zero-filled inputs.

## 2026-07-14 — E7b native-subset correction

- Select the retained electrodes, reference within that native montage, and
  pass only its signals and coordinates to REVE instead of retaining 64
  positions filled with zeros.
- The existing token-flattening probe cannot accept a variable token count.
  E7b therefore uses REVE's released fixed-dimensional attention pooling and a
  linear probe solely to validate the benchmark.
- Train on full-montage CAR and evaluate native 32/16/8 subsets under CAR/Cz,
  plus asymmetric region drops. Keep `native16@cz` as the primary view.
- A usable benchmark must pass the clean gate, show at least a 0.03 primary
  gap, remain at least 0.02 above chance, predict at least three classes, avoid
  a dominant class above 95%, and have normalized prediction entropy >=0.30.
- Lambda 10 is not evaluated in E7b because transplanting its fixed-width head
  would change the method. Montage-aware variable-set learning begins only
  after E7b validates the corrected stressor.

## 2026-07-15 — E7b result: native interface valid, readout gate failed

- The full-montage CAR BAcc was 0.3988, below the predeclared 0.45 clean gate.
  The released attention-pooled representation plus sklearn linear probe is
  therefore too weak to validate the downstream montage benchmark.
- Native 32/16/8 CAR BAcc values were 0.2616, 0.2489, and 0.2500. The primary
  `native16@cz` BAcc was 0.2606, below the 0.27 noncollapse threshold.
- The primary apparent full-to-native gap was 0.1382, but it cannot be treated
  as a useful stress effect because both the clean and noncollapse gates failed.
- CAR canonicalization removed reference residual exactly within each native
  montage, supporting the reference algebra and native channel/coordinate
  plumbing.
- Representation CKA decreased with retained channel count under Cz (about
  0.74 full, 0.64 at 32, 0.46 at 16, and 0.32 at 8). This is diagnostic only;
  it does not rescue the failed readout gate.
- Decision: do not repeat E7b or begin montage-aware training with this weak
  pooled feature. Build a stronger variable-cardinality token readout first.

## 2026-07-15 — E7c variable-set token readout protocol

- Keep REVE frozen and operate on its token set. A fixed bank of learned
  queries uses multihead attention to produce fixed-width output without
  flattening raw tokens or zero-filling missing channels.
- Predeclare query counts 4, 8, and 16. Select solely by CAR validation BAcc,
  with the smaller count breaking exact ties. Native test rows are unavailable
  during this selection.
- Use selected clean CAR test BAcc >=0.45 as a feasibility gate. Stop and push
  only the selection output if it fails.
- If it passes, freeze query count and evaluate the unchanged native 32/16/8
  and asymmetric region-drop suite under no defense and CAR canonicalization.
- Keep `native16@cz` primary and reuse E7b's hard-shift/noncollapse criteria.
  E7c validates the benchmark-compatible readout; it is not yet a claim for
  montage-aware representation learning.
- Only a usable E7c benchmark advances to montage dropout plus joint
  gauge/montage consistency over seeds 7/21/42.

## 2026-07-15 — E7c result: benchmark gate passed with a class-specific caveat

- q4 was selected by CAR validation BAcc 0.5595, strictly above q8 at 0.5561
  and q16 at 0.5548. The grid was close but not an exact tie, so the tie-break
  did not determine the selection.
- q4 reached clean full-CAR test BAcc 0.6112 and passed the 0.45 clean gate.
  `native16@cz` reached 0.3850; its full-to-native gap was 0.2263 with paired
  subject-bootstrap 95% CI [0.1790, 0.2724].
- Native CAR BAcc decreased monotonically over 32/16/8 channels: 0.4900,
  0.3844, and 0.3799. CAR canonicalization produced exactly equal CAR/Cz
  results within each fixed native montage.
- The predeclared noncollapse gate passed, but a post-hoc class audit found a
  functional two-class collapse at native16. Under CAR, left/right-fist recall
  was 0.772/0.762 while both-fists/both-feet recall was 0.000/0.004; 99.7% of
  predictions went to the first two classes.
- The collapsed classes retained AUROC 0.615 and 0.681, so ranking information
  remains and the decision failure may be recoverable through montage-aware
  training.
- Native16 CAR and Cz had nearly equal BAcc but disagreed on 11.5% of trials;
  native32 disagreement was 21.8%. Therefore, aggregate cancellation must not
  be described as absence of a reference effect.
- E7c remains a usable benchmark under its frozen rule. The functional-collapse
  criterion is a diagnostic target, not a post-hoc replacement gate.

## 2026-07-15 — E7d q4 reference/class-conditional closure protocol

- Freeze the validation-selected q4 architecture and CAR-only training. Add
  full-montage CAR/Cz/Pz/FCz evaluation without selecting anything from test.
- Require exact reproduction of E7c CAR predictions before accepting the new
  run, which prevents checkpoint variation from contaminating comparisons.
- Quantify full-reference BAcc gaps, prediction disagreement, per-class recall,
  per-class AUROC, and subject-bootstrap recall intervals.
- Compare full CAR to native16 CAR for montage loss, and native16 CAR to
  native16 Cz for the within-montage reference effect.
- Flag functional two-class collapse when the top two classes receive >=98% of
  predictions and at least two classes have recall <0.05. Preserve E7c's
  original gate and label this criterion post-hoc.
- If the collapsed classes retain AUROC >=0.55, proceed next to supervised
  montage dropout and a joint gauge/montage representation-consistency method.

## 2026-07-15 — E7d result: deterministic closure and recoverable class collapse

- The q4 CAR predictions matched E7c exactly, validating the E7d comparison.
- Full CAR-to-Cz/Pz/FCz BAcc gaps were 0.0373, 0.0452, and 0.0279, with
  prediction disagreements 19.2%, 27.1%, and 18.6%. Pz was the largest
  exploratory full-reference effect.
- Under Pz, left-fist recall fell from 0.632 to 0.435 while both-feet recall
  rose from 0.623 to 0.806; both subject-bootstrap intervals excluded zero.
  Per-class AUROC changed much less, supporting a relative class-margin bias
  rather than complete information destruction.
- Full CAR to native16 CAR lost 0.2268 BAcc and changed 60.9% of predictions.
  Both-fists/both-feet recall fell to 0.000/0.004 with bootstrap gaps 0.509 and
  0.618, while their AUROC remained 0.615/0.681.
- Native16 CAR versus Cz changed 11.5% of predictions but had a -0.0005 BAcc
  gap. Aggregate cancellation must not be interpreted as identical decisions.
- Hypothesis for the weak native Cz aggregate effect: sparse16 is a compact
  motor-centric montage containing Cz, so its internal CAR may be close to Cz.
  This is plausible but not established by E7d.

## 2026-07-15 — E7e native reference-geometry protocol

- Freeze q4 and CAR-only training. Evaluate complete CAR/Cz/Pz/Fz suites for
  native32 and native16, with no new hyperparameter or target selection.
- Pz and Fz are retained in both montages and are predeclared as less-central
  alternatives to Cz. Named electrodes are only a geometry diagnostic, not a
  continuous calibrated distance measure.
- Require exact reproduction of E7d CAR predictions before analysis.
- Report BAcc gaps, prediction disagreement, representation drift, class
  recall/AUROC, and paired subject-bootstrap recall intervals for every view.
- Support joint gauge/montage method scope if native16 Pz/Fz has absolute BAcc
  gap >=0.03 or absolute class-recall gap >=0.10 with CI excluding zero.
  Disagreement >=0.15 is supporting evidence only.
- If this suite-level criterion fails, make montage robustness the main method
  and keep gauge consistency as an auxiliary regularizer.

## 2026-07-15 — E7e result review and E8 calibration-control decision

- E7e reproduced q4 CAR predictions exactly and supported the predeclared
  joint gauge/montage scope. Native16 Pz changed BAcc from 0.3844 to 0.4205
  and moved both-feet recall from 0.004 to 0.300; native32 Fz reduced BAcc
  from 0.4900 to 0.4358.
- Aggregate BAcc masked larger class redistribution. Native32 Pz changed 38.2%
  of predictions while its aggregate BAcc moved only 0.0084; both-feet recall
  rose from 0.327 to 0.680 as the three other recalls fell.
- The user and assistant agreed not to jump directly to a representation loss.
  Stable per-class ranking makes validation-only calibration the necessary
  control before claiming a representation failure.
- E8 therefore freezes q4 and the E7e suite, saves raw validation/test logits,
  and compares identity, scalar temperature, class-bias, and vector scaling.
  Target-view-specific fitting is explicitly labeled an oracle known-view
  baseline; leave-one-view-out tests transfer to an unseen reference within a
  fixed montage.
- All calibration parameters are fitted only on validation subjects 71--89.
  Test subjects 90--109 remain untouched until final evaluation, and E8 must
  reproduce E7e identity predictions exactly before interpretation.
- The predeclared explanatory gate requires at least 50% reduction in the
  worst class-recall shift without more than 0.01 loss in worst native BAcc.
  Class-bias correction is the primary method matched to the observed margin
  shift; temperature is an argmax-invariant control and vector scaling is a
  sensitivity analysis, so no method is selected from test results. The result
  will determine whether to develop a calibration/readout method or proceed to
  joint gauge/montage representation learning.

## 2026-07-15 — E8 result and E9 reference-bias manifold decision

- All leakage and reproduction checks passed. View-specific class bias reduced
  the worst recall gap from 0.353 to 0.116 (67.3%) and increased worst native
  BAcc from 0.373 to 0.432, so the predeclared oracle calibration gate passed.
- Leave-one-view-out bias did not transfer. Its worst recall gap rose to 0.614,
  driven by native16 Pz both-feet recall 0.740 versus 0.126 at native16 CAR.
  Nevertheless, it significantly improved BAcc on native16 CAR/Cz/Fz and
  native32 Fz. Thus it is an unstable over-correction rather than a uniformly
  useless baseline.
- Vector scaling matched bias on decision stability while giving much lower
  ECE, so a final calibration method may need both scale and bias. E8 still
  predeclared bias as the primary causal diagnostic and did not select vector
  scaling post hoc.
- The limitation is now precise: a pooled correction can improve mean task
  performance yet amplify worst-reference class inconsistency because the
  required correction depends on reference identity and montage.
- E9 expands validation-only inference to every retained reference electrode.
  Subjects 71--80 fit oracle bias targets and subjects 81--89 evaluate them;
  reference identity is held out across native16 and native32 simultaneously.
- E9 compares global mean, E8-style pooled bias, nominal-topology ridge,
  label-free logit-statistics ridge, and their combination. Held-out target
  labels are available only to calculate oracle prediction error and an upper
  bound. A leakage unit test requires every deployable prediction to remain
  unchanged when those labels are perturbed.
- The combined predictor must beat the better simple bias baseline's RMSE by
  20%, reduce the worst recall gap by 30%, and lose no more than 0.01 mean
  BAcc versus the better simple baseline. Only then will an
  operator-conditioned calibrator be implemented.

## 2026-07-15 — E9 result review and E10 prior-stress decision

- E9 completed all 50 native validation views and 48 held-out target views.
  Topology, logit-only, and combined ridge all passed the predeclared gate.
  Their mean bias RMSE values were 0.223, 0.037, and 0.044 versus 0.559 for
  global mean and 0.562 for E8-style pooled bias.
- Logit-only/combined increased mean BAcc from 0.420 to 0.462 and reduced the
  worst class-recall gap from 0.478 to 0.205/0.181. Candidate fitting remained
  disjoint from held-out reference labels and PhysioNet test subjects.
- The E9 reproduction summary listed only native16/native32 CAR because view
  matching was case-sensitive. A manual case-normalized comparison verified
  all eight shared CAR/Cz/Fz/Pz views exactly: identical keys, predictions,
  logits, and maximum absolute logit difference 0.0. The audit code is fixed
  to normalize view case.
- E9's apparent logit manifold is largely an objective identity. For additive
  bias, supervised NLL uses labels only through empirical class proportions.
  The fit counts were 225/225/224/226. Replacing labels with that prior matched
  oracle bias at mean RMSE below 0.000001; using a label-free uniform prior
  still achieved RMSE 0.0035, mean BAcc 0.4629, and worst recall gap 0.181.
- The method limitation is now known-prior and batch dependence. With an
  unknown class mix, logit statistics confound reference bias with label shift;
  with a small batch, the prior-matching estimate has high variance.
- E10 is CPU-only on the committed E9 logits. It evaluates random batches from
  16 to 900 trials, balanced batches at 32/128, and 40%/70% controlled class
  skews at 128. Stress labels construct batches and score results only.
- The candidate shrinks prior-matching bias toward leave-one-electrode-out
  topology ridge. Its weight is estimated using nested errors from non-target
  references only. At primary random n=32 it must reduce RMSE by 20%, reduce
  mean maximum recall gap by 10%, preserve BAcc within 0.01, beat topology
  RMSE, and have a paired reference/resample-bootstrap RMSE interval below
  zero.
- A pre-push end-to-end CPU verification with the frozen defaults completed
  on the committed E9 logits. It produced 38.0% RMSE reduction, 19.5% mean
  maximum-recall-gap reduction, +0.0022 mean BAcc, and a paired RMSE-delta CI
  [-0.154, -0.069] at random n=32. Severe-skew RMSE was 3.0 times balanced
  RMSE at n=128. These temporary outputs are not committed; the user run is
  the independent reproduction artifact.

## 2026-07-16 — E10 reproduction review and E11 decision

- The user's committed E10 output reproduced the pre-push result: at random
  `n=32`, topology shrinkage reduced bias RMSE from 0.2967 to 0.1839 (38.0%),
  reduced mean maximum recall gap from 0.1168 to 0.0941 (19.5%), and increased
  mean BAcc by 0.0022. The paired RMSE-delta interval was [-0.154, -0.069].
- Severe 70%-skew RMSE at `n=128` was 0.3481 versus 0.1161 for balanced
  batches, a ratio of 3.00. This confirms that E10's batch-size-only weight
  cannot distinguish observation-operator bias from target label shift.
- E11 separates prior-model subjects 71--75 from adaptation subjects 76--80,
  estimates a soft confusion matrix using leave-one-subject-out CAR
  predictions, and performs regularized pseudo-prior inversion anchored to the
  nominal prior. Subjects 81--89 remain evaluation-only; PhysioNet test
  subjects remain unused.
- The deployable candidate uses frozen target logits, source soft confusion,
  and leave-one-electrode-out topology only. Adaptation labels construct and
  audit controlled stress batches but cannot fit any candidate quantity. A
  perturbation test enforces this invariance.
- The gate is intentionally two-level. Mean severe robustness requires a 5%
  RMSE reduction with a paired interval below zero and preserved task metrics.
  The strict method claim additionally requires every dominant-class direction
  to improve.

## 2026-07-16 — E11 pre-push CPU verification

- The frozen default run completed in about 80 seconds. At severe 70% skew and
  `n=128`, operator-confusion shrinkage reduced mean RMSE from 0.1856 to 0.1690
  (8.95%); the paired 95% delta interval was [-0.0219, -0.0113]. It also
  reduced mean maximum recall gap from 0.0990 to 0.0937, with mean BAcc changing
  by -0.0007.
- Random `n=32` RMSE changed from 0.1771 to 0.1753, and balanced `n=128` RMSE
  changed from 0.1439 to 0.1391, so both nominal-preservation checks passed.
- Three dominant-class directions improved. Right-fist-dominant RMSE worsened
  slightly from 0.1452 to 0.1482 (+0.0031, about 2.1%). Accordingly, mean
  severe robustness is supported, class-uniform robustness is false, and the
  strict method gate remains false.
- The regularized pseudo-prior's severe-shift RMSE to the true batch prior was
  0.254, so E11 must not claim accurate label-prior recovery. The defensible
  interpretation is partial correction from frozen logits.
- Decision: preserve the failed strict gate and make the next step a
  class-conditional safeguard, with the right-fist failure as the explicit
  regression case. Do not tune a post-hoc global threshold merely to make E11
  pass.

## 2026-07-16 — E11 reproduction review and strict-split correction

- The user pushed a complete E11 output suite. Its headline result reproduced
  the pre-push run, including the right-fist regression and failed strict gate.
- Protocol review found a stronger limitation: the probability model used
  subjects 71--75 and adaptation used 76--80, but topology used 71--80. Thus
  subjects 76--80 influenced both topology and adaptation. The archived E11
  result is exploratory and cannot support a strict cross-subject claim.
- The E11 validator and defaults now require topology/prior subjects 71--75,
  adaptation subjects 76--80, and evaluation subjects 81--89. Adaptation and
  evaluation overlap is also rejected explicitly.
- Under this strict split, E11's operator candidate improved all four severe
  raw class directions versus fixed shrinkage, but its severe mean RMSE was
  0.2797, worse than static topology RMSE 0.2239. Therefore pseudo-prior
  correction was too aggressive even after E11's scalar shrinkage.

## 2026-07-16 — E12 source-only class/operator safeguard

- E12 learns one trust cap per output class and held-out reference identity in
  full four-class zero-sum bias space. It uses source subjects 71--75 only.
- The outer target reference is removed from all source examples. For every
  source pseudo-target, both its reference and the outer target reference are
  removed from the nested topology fit. No target-reference adaptation or
  evaluation label fits a cap.
- At deployment, the class caps are upper bounds on E11's scalar trust, so the
  method can move an uncertain update back toward topology but cannot amplify
  it. This directly addresses the observed over-correction rather than tuning
  a global target threshold post hoc.
- The frozen default CPU run completed end to end. Mean learned caps were
  0.2110 (left fist), 0.2589 (right fist), 0.5768 (both fists), and 0.2323
  (both feet); every controlled source condition improved over topology in all
  outer folds.
- At random `n=32`, RMSE changed from 0.2364 to 0.1983. At balanced `n=128`,
  it changed from 0.3036 to 0.1927. Both preservation gates passed.
- At severe 70% skew, RMSE changed from 0.3489 to 0.2013 (42.3% reduction) and
  beat topology-only RMSE 0.2239. Mean BAcc changed from 0.46225 to 0.46119;
  mean maximum recall gap improved from 0.1360 to 0.1049.
- All four dominant-class raw and reference-clustered point directions
  improved. Left fist, both fists, and both feet had intervals below zero.
  Right fist improved by point estimate, but its clustered interval was
  approximately [-0.0266, 0.0206]. No harm was detected, but its improvement
  was not individually confirmed.
- Decision: E12 passes the predeclared gate for repeated-seed confirmation but
  not the paper-level class-uniform gate. The next user artifact must contain
  the complete `outputs/reve_set_class_safeguard_audit_s7` directory, including
  `strict_prior_baseline`. If independently reproduced, proceed to repeated
  seeds and an external open EEG dataset rather than further seed-7 tuning.

## 2026-07-16 — E12 reproduction review and E13 decision

- The user pushed all 16 expected E12 artifacts, including the nested strict
  E11 control. The committed run reproduced the strict 71--75 / 76--80 /
  81--89 split and all split-disjointness checks.
- E12 RMSE was 0.1983 at random `n=32`, 0.1927 at balanced `n=128`, and
  0.2013 under mean severe skew. The corresponding strict E11 values were
  0.2093, 0.2407, and 0.2797; topology-only RMSE was 0.2239.
- Review identified a comparison limitation in the original E12 gate: it used
  fixed E10 shrinkage as its paired class baseline even when strict E11 or
  topology was stronger. E13 is a no-refit, post-hoc audit against both strong
  controls using paired repeat-by-reference bootstrap intervals.
- The E13 pre-push run found mean E12 point improvements against both strong
  baselines in random, balanced, and severe regimes. The primary delta versus
  strict E11 was -0.0076 with interval [-0.0193, 0.0033], so a single-seed
  strong-baseline mean confirmation remains false.
- Right-fist-dominant E12 was worse than strict E11 by 0.0182 RMSE, with
  interval [0.0006, 0.0361]. Left-fist and both-fists also lacked intervals
  below zero against topology, although no harm was detected for those two.
- Decision: allow genuinely new probe seeds for a mean-only hypothesis, but do
  not advance the current class-uniform method or paper claim. Since the E13
  rule follows seed-7 inspection, it is falsification-only; it cannot convert
  the existing result into confirmatory evidence.

## 2026-07-16 — E13 reproduction and E14 protocol repair

- The user pushed commit `f1699fa` with exactly the three expected E13 files
  and no code changes. The results reproduced the pre-push audit numerically.
- Mean E12 RMSE improved against strict E11 and topology under random,
  balanced, and severe conditions. The primary strict-E11 interval crossed
  zero, while balanced and severe intervals were below zero against both
  baselines. Mean task-metric noninferiority passed.
- The current class-uniform method remains rejected. Right-fist-dominant RMSE
  was worse than strict E11 by 0.0182 with interval [0.0006, 0.0361]. The
  both-fists recall-gap comparison against topology also failed the frozen
  noninferiority check.
- Before launching repeated seeds, code review found that the old q4 probe used
  subjects 71--89 for early stopping and E12 reused 71--89 for its downstream
  source/adaptation/evaluation audit. This preserves fairness of the paired
  E12/E11/topology comparison, but makes seed 7 exploratory rather than
  independently confirmatory.
- E14 introduces a four-way split: probe train 1--60, probe validation 61--70,
  audit 71--89, and reserved test 90--109. A validation-only run mode emits
  audit logits without fitting or scoring test subjects.
- The frozen E13 gate is applied only to untouched probe seeds 21 and 42.
  Aggregation bootstraps probe seed, batch repeat, and held-reference identity.
  A pass advances only the mean-method hypothesis to an external open dataset;
  class-uniform evidence cannot be restored by E14.

## 2026-07-16 — E14 seed-42 known-prior optimizer repair

- The user pushed complete E14 audit logits for untouched probe seeds 21 and
  42, plus the completed strict E12 audit for seed 21. Seed 42 reached the CPU
  audit but stopped inside `fit_known_prior_bias`.
- Reproduction on the committed seed-42 CSV localized the first failure to
  `native16@Cz`, a random `n=16` batch, repeat 0. The soft-confusion prior was
  `[0, 0.610154, 0.200852, 0.188993]`, so one valid estimated class mass was
  exactly zero.
- The Newton solver's objective already had positive L2 regularization, which
  makes this solution finite. However, a legacy `[-5, 5]` hard clip forced the
  first free bias to `-5`; the clipped Armijo step could not satisfy the
  unconstrained gradient tolerance and produced a false line-search failure.
- The hard clip was removed while retaining stable log-sum-exp evaluation,
  stable softmax, positive L2, and backtracking. A regression test now requires
  a zero-mass class to converge beyond the old boundary.
- The entire seed-42 E12 pipeline was replayed on the exact 88,435-row committed
  prediction file and completed without another optimization failure. The
  user's existing seed-21 audit and both trained probe-logit runs remain
  reusable; only the missing seed-42 E12 audit and final E14 aggregation need
  to run after pulling the fix.

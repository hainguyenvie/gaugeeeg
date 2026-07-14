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

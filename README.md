# GaugeEEG

GaugeEEG is a feasibility-first benchmark for a concrete question:

> Do EEG representations remain stable when the same underlying recording is
> expressed under a different, physically valid voltage reference?

The project transfers the *scientific-rule-informed self-supervision* idea from
NAI-SSL to a different problem rather than treating NAI-SSL as a baseline. The
anchor baseline is [REVE](https://brain-bzh.github.io/reve/), a NeurIPS 2025 EEG
foundation model with public code, released weights, and a published
PhysioNetMI evaluation. The first benchmark uses the open-access
[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/).

## What is implemented

- Exact EEG re-referencing operators: common average, single-electrode, and
  deterministic linear reference.
- A synthetic test proving that CAR canonicalization removes the additive
  reference (gauge) component to numerical precision.
- Automatic download and preprocessing of PhysioNetMI through MNE.
- Leakage-safe subject splits and a four-class motor-imagery task matching the
  task definition used by REVE (left fist, right fist, both fists, both feet).
- A lightweight log-bandpower baseline that runs without a GPU or model access.
- Frozen REVE embeddings plus both a lightweight logistic probe and an
  official-like token-level PyTorch probe.
- Reference-shift accuracy, balanced accuracy, macro-F1, AUROC, cosine drift,
  relative L2 drift, and linear CKA.
- Two initial conditions: unprotected input and exact CAR canonicalization.

This is deliberately a pilot, not yet a claim of novelty. The first goal is to
measure whether the limitation is large and reproducible enough to support a
paper.

## Quick start: CPU feasibility run

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/hainguyenvie/gaugeeeg.git
cd gaugeeeg

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[data,dev]"

python -m unittest discover -s tests -v
gaugeeeg synthetic
gaugeeeg run --config configs/smoke.yaml --encoder bandpower
gaugeeeg run --config configs/pilot.yaml --encoder bandpower
```

MNE downloads the selected EDF files from PhysioNet on first use and reuses
them afterward. The smoke run uses 3 subjects and two runs (one unilateral and
one bilateral, covering all four labels) to validate the complete pipeline;
the pilot uses 12 subjects and all six motor-imagery runs for a first
effect-size estimate. Their generated files are written to
`outputs/smoke_bandpower/` and `outputs/pilot_bandpower/`.

## Frozen REVE experiment

REVE-base is a gated Hugging Face model. Before the first run:

1. Open [brain-bzh/reve-base](https://huggingface.co/brain-bzh/reve-base), read
   the responsible-use agreement, and request access.
2. Install the REVE dependencies and authenticate locally.

```bash
pip install -e ".[data,reve,dev]"
hf auth login

gaugeeeg run \
  --config configs/pilot.yaml \
  --encoder reve \
  --device cuda \
  --output-dir outputs/pilot_reve
```

Use `--device cpu` if no CUDA GPU is available; it will be slower. REVE is kept
frozen. The implementation follows the released interface: 200 Hz EEG,
electrode coordinates from `brain-bzh/reve-positions`, input in microvolts
divided by 100, and the model's attention-pooling head.

## Next experiment: reproduce the clean REVE baseline first

The 12-subject pilot is not sufficient for a reference-robustness claim: its
clean REVE balanced accuracy is near four-class chance. Before running any more
reference views, reproduce a useful clean classifier on the released 70/19/20
subject split:

```bash
gaugeeeg run --config configs/reve_clean_gate.yaml --device cuda
cat outputs/reve_clean_gate/summary.json
```

This configuration follows the released REVE PhysioNetMI LP shape: frozen
token features, pretrained query-token initialization, query attention,
flattened tokens, RMSNorm, dropout, and a linear head trained for up to 20
epochs. It uses AdamW as the local equivalent of REVE's StableAdamW setting.
Token caches are stored as float16 because the full frozen feature tensor is
large. Expect several GB of cache and substantial first-run GPU time.

`clean_gate_passed` requires balanced accuracy >= 0.45. The paper reports
0.537 +/- 0.005 for pooled REVE-Base and 0.510 +/- 0.012 for its non-pooled
variant; the gate is deliberately below those targets. Do not interpret
reference shifts until this gate passes.

## Full reference benchmark

The clean gate has passed on the full split (test balanced accuracy 0.5567).
Run the first reference-stress screening with the same token-level probe:

```bash
gaugeeeg run \
  --config configs/reve_reference_stress.yaml \
  --device cuda
```

This evaluates CAR, Cz, Pz, FCz, and a deterministic random linear reference
with one probe trained only on CAR. It reuses the clean-gate train, validation,
and CAR-test token cache, so only the shifted test views require new REVE
inference. Do not use `--force-recompute`, because that bypasses cache reuse.

The screening hypothesis is supported when `stress_effect_detected` is true,
corresponding to at least a 0.03 absolute balanced-accuracy drop for one valid
reference. The run also saves `probe_history_none.csv` and a local best-probe
checkpoint. If the screen is positive, the next stage repeats the experiment
with seeds 7, 21, and 42.

## Statistical audit after the 3-seed screen

The first three-seed screen found a marginal effect: Cz reduced balanced
accuracy by about 3.2 points on average, while the worst view and threshold
decision changed across probe seeds. Run the subject-level audit before making
a downstream-robustness claim:

```bash
gaugeeeg run --config configs/reve_statistical_audit.yaml \
  --probe-seed 7 --output-dir outputs/reve_statistical_audit_s7

# Re-run seed 7 once to check strict reproducibility.
gaugeeeg run --config configs/reve_statistical_audit.yaml \
  --probe-seed 7 --output-dir outputs/reve_statistical_audit_s7_repeat

gaugeeeg run --config configs/reve_statistical_audit.yaml \
  --probe-seed 21 --output-dir outputs/reve_statistical_audit_s21

gaugeeeg run --config configs/reve_statistical_audit.yaml \
  --probe-seed 42 --output-dir outputs/reve_statistical_audit_s42

gaugeeeg aggregate-audit \
  --runs outputs/reve_statistical_audit_s7 \
         outputs/reve_statistical_audit_s21 \
         outputs/reve_statistical_audit_s42
```

All runs fix `reference_seed=7`, reuse the existing seed-7 token cache, and
change only probe initialization/data order. Do not pass `--force-recompute`.
The audit saves trial predictions, per-subject metrics, and a paired
subject-cluster bootstrap confidence interval. Aggregation uses sample standard
deviation (`ddof=1`) and selects one fixed worst reference across seeds instead
of averaging a separately selected maximum from each seed.

## Class-bias audit and held-out-reference method screen

The deterministic audit showed that aggregate BAcc hides a much larger and
repeatable class-conditional shift. Generate the class-level evidence without
any GPU work:

```bash
gaugeeeg class-bias-audit \
  --runs outputs/reve_statistical_audit_s7 \
         outputs/reve_statistical_audit_s21 \
         outputs/reve_statistical_audit_s42 \
  --output-dir outputs/reve_class_bias_audit
```

The predeclared method target is the Cz-induced left-fist recall gap. Cz is
held completely outside the training views. First run supervised multi-view
augmentation on CAR/Pz/FCz:

```bash
gaugeeeg run \
  --config configs/reve_consistency_screen.yaml \
  --probe-objective multi_view_ce \
  --output-dir outputs/reve_multiview_ce_s7
```

The first command must extract Pz/FCz train and validation tokens, so it is the
expensive run. It reuses all existing CAR and test tokens. Next run the proposed
rule-consistency objective; this should reuse every frozen token cache:

```bash
gaugeeeg run \
  --config configs/reve_consistency_screen.yaml \
  --probe-objective rule_consistency \
  --consistency-weight 1.0 \
  --output-dir outputs/reve_rule_consistency_s7
```

Compare both methods against the deterministic CAR-only seed-7 baseline:

```bash
gaugeeeg compare-methods \
  --baseline outputs/reve_statistical_audit_s7 \
  --augmentation outputs/reve_multiview_ce_s7 \
  --consistency outputs/reve_rule_consistency_s7 \
  --output-dir outputs/reve_consistency_comparison_s7
```

The method screen passes when the held-out-Cz left-fist recall gap is reduced
by at least 30% while CAR BAcc falls by no more than one point. A contribution
from the rule loss is supported only if `rule_consistency` also beats
`multi_view_ce`; otherwise recovery must be attributed to ordinary data
augmentation. Do not run seeds 21/42 until the seed-7 method screen passes.

## Multi-seed confirmation after the seed-7 pass

The seed-7 screen passed, but the direct paired advantage of consistency over
augmentation was still uncertain across test subjects. Run the predeclared
confirmation with one command:

```bash
git pull
source .venv/bin/activate
make consistency-multiseed
```

Equivalently, run `bash scripts/run_consistency_multiseed.sh`. Set
`DEVICE=cuda:1` before the command if the experiment should use another GPU.
The script intentionally does not pass `--force-recompute`: it reuses the
frozen REVE feature cache and trains only four new probes (two objectives for
seeds 21 and 42). It also regenerates the seed-7 comparison with a direct
paired-method bootstrap.

The final decision is written to:

```text
outputs/reve_consistency_comparison_multiseed/aggregate_method_summary.json
```

Interpret `rule_loss_evidence_status` as follows:

- `supported`: every consistency run passes the recovery/clean gate and the
  hierarchical 95% CI for its recall-gap advantage over augmentation is above
  zero.
- `promising_but_inconclusive`: the mean advantage is positive, but its CI
  crosses zero.
- `not_supported`: the consistency objective fails the gate or has no positive
  mean advantage. In this case, attribute robustness to multi-view
  augmentation rather than the rule loss.

After the run, commit or send back the six new method run directories, the
three per-seed comparison directories, and
`outputs/reve_consistency_comparison_multiseed/`. Do not run a consistency
weight sweep until this confirmation is interpreted.

## Validation-only consistency-weight ablation

The three-seed confirmation was directionally stable but statistically
inconclusive: consistency beat augmentation on the primary metric in all three
seeds, while the hierarchical 95% CI narrowly crossed zero. The next stage
tests whether `lambda=1` underweights the rule term without tuning on Cz.

```bash
git pull
source .venv/bin/activate
make consistency-lambda-ablation
```

The candidate grid is `lambda in {0, 0.3, 1, 3, 10}`. Lambda zero is the
existing multi-view CE run and lambda one is the existing consistency run, so
the script trains only nine new probes (`0.3/3/10 x seeds 7/21/42`). Frozen
REVE features are reused and `--force-recompute` is never passed.

Selection is test-blind: lambda is chosen by mean validation balanced accuracy
across CAR/Pz/FCz and the three probe seeds, with the smaller lambda breaking
an exact tie. Only after that choice is fixed does the analyzer read held-out
Cz predictions and run the hierarchical paired bootstrap. Test results for
the complete grid are saved as transparent ablations and must not be used to
reselect lambda.

The primary output is:

```text
outputs/reve_consistency_lambda_ablation/lambda_ablation_summary.json
```

Send back or commit the nine `reve_rule_consistency_lam*_s*` directories and
`outputs/reve_consistency_lambda_ablation/`. If validation selects lambda zero,
the rule loss is not supported. If it selects a positive lambda, interpret only
the nested `selected_vs_augmentation` evidence, not the best-looking Cz row.

## Reference-plus-sparse-montage feasibility screen

Validation selected `lambda=10`, and its held-out-Cz advantage over ordinary
multi-view CE passed the predeclared hierarchical bootstrap. The next screen
asks whether that full-montage reference rule transfers when electrodes are
also unobserved:

```bash
git pull
source .venv/bin/activate
make test
make montage-screen
```

The runner first verifies that the preceding lambda summary exists, that
lambda 10 was selected without the target view, and then evaluates four fixed
seed-7 methods: CAR-only, CAR canonicalization, multi-view CE, and the selected
rule-consistency probe. The OOD suite contains nested motor-centric 32/16/8
channel observations under CAR and Cz plus left/right motor-region dropout.

Sparse observations use the explicit convention `montage@reference`: apply
the physical reference first, then zero-fill unobserved channels while keeping
the original channel order. This lets the same frozen REVE encoder and token
probe consume every paired trial. It does not reconstruct missing signals, and
CAR canonicalization is no longer an exact inverse once channels are absent.

The primary view is fixed as `sparse16@cz`; do not choose a different montage
after reading the output. Interpret this one-seed stage only as a feasibility
screen. The decision artifact is:

```text
outputs/reve_montage_screen_s7/montage_screen_summary.json
```

Commit the four `outputs/reve_montage_*_s7` run directories and
`outputs/reve_montage_screen_s7/`. If the CAR-only primary BAcc gap is at least
three points, the next stage should add montage-aware training rather than
immediately repeating this unchanged method over more seeds.

## Corrected native channel-subset screen

E7a confirmed that zero-filling missing electrodes is not a valid proxy for
REVE's native arbitrary-layout interface: nearly every sparse condition
collapsed to one predicted class. E7b removes each missing signal *and* its
position before REVE, then uses the model's released attention pooling to keep
the downstream feature dimension fixed:

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make native-montage-screen
```

Use `DEVICE=cuda:1` if required. This run intentionally does not reuse the
fixed-width token-flattening head or lambda 10: both assume a constant token
count and therefore cannot test native subsets without changing the method.
It first validates the benchmark using a full-montage CAR linear probe and a
CAR-canonicalization control.

The primary target remains fixed at 16 observed channels under Cz. A usable
benchmark must pass the clean gate, lose at least 0.03 BAcc on the primary
target, and avoid the E7a collapse signature: at least three predicted classes,
normalized prediction entropy at least 0.30, largest predicted-class share at
most 0.95, and primary BAcc at least 0.02 above chance. The decision is saved
to:

```text
outputs/reve_native_montage_screen_s7/native_montage_screen_summary.json
```

The E7b run did not pass its clean gate: released attention pooling plus a
linear probe reached only 0.3988 clean CAR BAcc, and `native16@cz` was 0.2606.
This validates the native subset plumbing but leaves the benchmark outcome
confounded by a weak readout.

## Variable-set token readout gate

E7c replaces only that readout. It uses learned queries with multihead
attention to pool a variable number of frozen REVE tokens into a fixed-width
classification input. Query count is selected from `{4, 8, 16}` using CAR
validation BAcc; native-montage results do not exist during selection.

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make set-native-screen
```

The script reuses existing full-CAR token caches. It first writes
`outputs/reve_set_head_selection_s7/set_head_selection.json`. If the selected
head does not reach clean CAR test BAcc 0.45, it stops before the expensive
native extraction. If it passes, it freezes the selected query count, runs the
same native 32/16/8 and region-drop suite under no defense and CAR
canonicalization, then writes:

```text
outputs/reve_set_native_montage_screen_s7/native_montage_screen_summary.json
```

E7c is a readout-validity stage, not yet the proposed montage-aware method. A
usable native benchmark is the prerequisite for the next comparison between
full-CAR training, montage dropout, and joint gauge/montage consistency.

E7c passed every predeclared gate at seed 7. Validation selected q4 with BAcc
0.5595 and the clean CAR test reached 0.6112. The primary `native16@cz` BAcc
was 0.3850, producing a subject-bootstrap gap of 0.2263 with 95% interval
[0.1790, 0.2724]. Post-hoc prediction inspection found a more specific
failure: the 16-channel model retained useful AUROC but almost never selected
the bilateral classes.

## q4 reference and class-conditional closure

E7d adds the missing full-montage `Cz/Pz/FCz` rows for the frozen q4
architecture and formalizes the class-collapse diagnostic. It does not select
another query count or alter E7c's validity rule:

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make set-reference-closure
```

The analyzer requires the new full-reference run to reproduce E7c's CAR
predictions exactly. It then reports paired BAcc/disagreement, per-class
recall/AUROC, subject-bootstrap recall intervals, effective predicted classes,
and the native16 CAR-versus-Cz redistribution. Push these outputs after the
run:

```text
outputs/reve_set_full_reference_q4_none_s7
outputs/reve_set_reference_closure_s7
```

E7d reproduced q4 CAR predictions exactly. Full-montage Pz was the largest
aggregate shift (0.0452 BAcc gap and 27.1% prediction disagreement), while the
native16 montage gap was 0.2268. Full-reference class recall moved far more
than per-class AUROC, supporting a relative class-margin bias; native16 caused
a functional two-class collapse while retaining above-chance bilateral-class
ranking.

## Native reference-geometry diagnostic

E7e tests whether the near-zero native16 CAR-versus-Cz BAcc gap generalizes to
references away from the montage center. The complete predeclared suite uses
Cz, Pz, and Fz within both 16- and 32-channel native montages; it does not pick
a best test reference:

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make set-reference-geometry
```

The run must reproduce E7d's CAR predictions exactly. The suite supports a
joint gauge/montage method scope only if native16 Pz/Fz produces either an
absolute BAcc gap of at least 0.03 or an absolute class-recall gap of at least
0.10 whose subject-bootstrap interval excludes zero. Push:

```text
outputs/reve_set_native_reference_geometry_q4_s7
outputs/reve_set_reference_geometry_audit_s7
```

## Reading the output

Each run creates:

- `metrics.csv`: one row per defense and test reference.
- `summary.json`: compact best/worst gaps and run metadata.
- `resolved_config.yaml`: the exact configuration used.
- `feature_cache/`: reusable frozen features for interrupted or repeated runs.
- `predictions.csv`: aligned trial predictions and class probabilities when
  enabled by the statistical-audit config.
- `subject_metrics.csv`: one row per test subject and reference view.
- `paired_subject_bootstrap.csv`: paired subject-cluster bootstrap intervals.

The primary stress-test quantity is:

```text
reference_gap = balanced_accuracy(CAR test) - balanced_accuracy(shifted test)
```

The pilot is promising if the unprotected frozen REVE model has a repeatable
drop of at least 3 percentage points for one or more valid references while a
rule-informed defense recovers most of the loss. If REVE is already stable
(less than 1 point drop and cosine similarity above 0.98), the planned pivot is
to harder but still lawful convention shifts: bipolar derivations, missing
electrodes, and cross-dataset montage transfer.

See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) for the hypothesis,
research gap, contribution ladder, and falsification criteria. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the exact experiment sequence.
See [conversation.md](conversation.md) for the chronological evidence and
decisions from each research iteration.

## Reproducibility notes

- Splits are by subject, never by trial.
- The classifier is selected on the CAR validation split, then refit on
  train+validation and evaluated once on each test reference.
- A deterministic seed defines the linear-reference weights.
- Downloaded recordings, model weights, and generated features are ignored by
  Git.
- Model results should not be compared directly with REVE's paper table until
  the full configuration and repeated seeds are used.

## External resources and licenses

- REVE code is MIT licensed; its model weights have a separate responsible-use
  agreement and must not be redistributed by this repository.
- PhysioNet EEGMMIDB files are available under the Open Data Commons
  Attribution License v1.0.
- GaugeEEG code is released under the MIT License.

## Validation-only calibration control

E8 tests the alternative explanation raised by E7d/E7e: reference changes may
move relative class margins while preserving most within-class ranking. It
fits no parameters on test subjects. Temperature, class-bias, and vector
scaling are fitted on subjects 71--89 and evaluated on subjects 90--109 under
both target-view-specific and leave-one-view-out protocols:

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make set-calibration-control
```

The runner requires the existing E7c q4 selection and E7e predictions. It
reproduces the deterministic q4 experiment while saving raw validation/test
logits, verifies identity predictions against E7e, then writes:

```text
outputs/reve_set_calibration_logits_q4_s7
outputs/reve_set_calibration_control_s7
```

Push both directories. `calibration_summary.json` distinguishes an oracle
known-view calibration result from leave-one-view-out generalization; scalar
temperature is retained as an argmax-invariant NLL/ECE control.

## Validation-only reference-bias manifold

E9 asks whether the target-specific correction exposed by E8 can be predicted
without labels from the held-out reference. It evaluates every electrode in
the native16 and native32 montages on validation subjects only. Subjects
71--80 fit per-reference bias targets; subjects 81--89 evaluate leave-one-
electrode-out transfer. The same electrode identity is removed from both
montages in each fold:

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make set-bias-manifold
```

The controls are global-mean and pooled bias (the E8 strategy on the expanded
grid). Candidate predictors use nominal 10--20 topology, label-free batch-logit
statistics, or their combination. Target-reference labels are used only for
the oracle upper bound and prediction-error audit, never for fitting a
candidate. A candidate must beat the better simple control. E9 does not
evaluate the reference grid on PhysioNet test subjects. Push:

```text
outputs/reve_set_bias_manifold_logits_q4_s7
outputs/reve_set_bias_manifold_audit_s7
```

## Known-prior and small-batch stress audit

E10 resolves the main E9 ambiguity. For additive class bias, supervised NLL
uses labels only through their empirical class proportions. A known balanced
task prior can therefore replace labels in a convex prior-matching objective.
E10 first verifies that identity, then asks whether nominal electrode topology
is useful when only a small unlabeled target batch is available:

```bash
git pull
source .venv/bin/activate
make test
make set-prior-stress
```

This stage is CPU-only and reuses the committed E9 validation logits. It tests
random batches of 16--900 trials, balanced batches at the primary and stress
sizes, and controlled 40%/70% single-class skews. Labels construct the
balanced/skew stress batches and score the audit, but never fit a deployable
target correction.

The candidate shrinks known-prior bias toward a leave-one-electrode-out
topology prediction. Its mixing weight is estimated only from non-target
references. The predeclared primary condition is a random batch of 32 trials.
Push:

```text
outputs/reve_set_prior_stress_audit_s7
```

The decision file is `prior_stress_summary.json`. It distinguishes a genuine
small-batch benefit from the full-batch prior-matching identity and explicitly
reports failure under class-prior confounding.

## Cross-subject prior-identifiability audit

E11 addresses the limitation confirmed by E10: fixed topology shrinkage helps
at small batch sizes, but a target batch's class mix remains confounded with
reference-induced class bias. It tests whether a source-trained soft confusion
operator can extract a conservative pseudo-prior from frozen target logits and
use it to correct severe prior shift.

```bash
git pull
source .venv/bin/activate
make test
make set-prior-identifiability
```

This stage is CPU-only and reuses the committed E9 validation logits. The
class-probability and topology models use subjects 71--75; target adaptation
batches use disjoint subjects 76--80; task effects are audited on subjects
81--89. Its confusion matrix is estimated from leave-one-subject-out source
predictions. A held-out electrode identity is still excluded across both
montages. The current command enforces all three subject groups as disjoint.

Labels from target adaptation subjects only construct the random, balanced,
and controlled-skew batches and audit true prior error. They do not fit the
pseudo-prior, candidate bias, or mixing weight. E11 separately reports:

- mean severe-skew robustness;
- class-uniform severe-skew robustness across all four dominant classes; and
- the strict method gate, which requires both plus nominal preservation.

The pre-push reference run reduced mean severe-skew bias RMSE by 8.95% with a
paired 95% interval below zero, while preserving the nominal conditions. It
improved three of four dominant-class directions; right-fist-dominant RMSE
increased by 0.0031. The strict gate therefore remains false. This is an
intentional falsification result and motivates a class-conditional safeguard,
not a post-hoc relaxation of the gate.

Push the complete output directory after running:

```text
outputs/reve_set_prior_identifiability_audit_s7
```

The primary decision file is `prior_identifiability_summary.json`. Expected
runtime for the reference CPU run was about 80 seconds, excluding tests.

The archived first E11 result used subjects 76--80 in both topology fitting and
adaptation. Its 8.95% mean result remains useful as an exploratory diagnostic,
but not as strict cross-subject evidence. E12 reruns E11 internally with the
corrected 71--75 / 76--80 / 81--89 split before evaluating the new safeguard.

## Source-only class/operator trust safeguard

E12 addresses two E11 failures without tuning on target labels: strict
topology/adaptation separation and a harmful class-specific update. For each
held-out electrode identity, it learns four diagonal trust caps in the
zero-sum class-bias space using only subjects 71--75. The target electrode is
excluded from both the source examples and each nested topology fit. At
deployment, these caps can only reduce E11's existing pseudo-prior weight.

```bash
git pull
source .venv/bin/activate
make test
make set-class-safeguard
```

This stage is CPU-only and takes roughly 3--5 minutes on the reference machine.
It writes both E12 results and the corrected strict E11 control beneath:

```text
outputs/reve_set_class_safeguard_audit_s7
outputs/reve_set_class_safeguard_audit_s7/strict_prior_baseline
```

The pre-push run reduced severe-skew bias RMSE from 0.3489 to 0.2013 (42.3%),
beating topology-only RMSE 0.2239. All four dominant-class point estimates
improved and no class-specific harm was detected. The right-fist clustered
interval still crossed zero, so `paper_level_class_uniform_claim_supported`
remained false. This licenses repeated-seed and external-dataset confirmation,
not a paper-level class-uniform claim.

Push the complete output directory. The decision file is
`class_safeguard_summary.json`; do not omit its `strict_prior_baseline`
subdirectory.

## Post-hoc strongest-baseline audit

E13 corrects an overly weak comparison in the original E12 gate. It does not
refit a model. Instead, it reuses the complete E12 metrics and applies a paired
repeat-by-reference bootstrap against both strict E11 operator-confusion
shrinkage and the static topology-only predictor:

```bash
git pull
source .venv/bin/activate
make test
make set-strong-baseline-audit
```

The audit is CPU-only and normally finishes in seconds after E12 exists. The
pre-push screen found that E12 had lower mean RMSE than both strong baselines in
random, balanced, and severe-skew regimes. At the primary random `n=32`
condition, however, its RMSE delta versus strict E11 was -0.0076 with a paired
95% interval of [-0.0193, 0.0033], so the single-seed mean result is not yet
confirmed. More importantly, right-fist-dominant severe RMSE was 0.0182 higher
than strict E11, with interval [0.0006, 0.0361]. The mean-only method is
therefore eligible for genuinely new-seed confirmation, while the
class-uniform claim is rejected for the current method.

Because this stronger gate was defined after inspecting seed 7, E13 can
falsify a claim but cannot confirm a paper claim. Push the three-file output:

```text
outputs/reve_set_strong_baseline_audit_s7
```

The decision file is `strong_baseline_summary.json`. Do not rerun or modify
E12 while producing this audit.

## Untouched-probe-seed mean-method confirmation

E14 repairs one more protocol limitation before spending compute on repeated
seeds. The old q4 probe used subjects 71--89 for early stopping, while E12
later divided the same subjects into source, adaptation, and evaluation sets.
This does not invalidate the within-run comparison among E12, E11, and
topology, but it prevents seed 7 from serving as independent confirmation.

E14 freezes the E13 rule and uses a four-way subject split:

- probe training: subjects 1--60;
- probe early stopping: subjects 61--70;
- downstream source/adaptation/evaluation audit: subjects 71--89; and
- reserved PhysioNet test: subjects 90--109, never fitted or scored.

Run the two untouched default probe seeds, 21 and 42:

```bash
git pull
source .venv/bin/activate
make test
make set-probe-seed-confirmation
```

Set `DEVICE=cuda:1` to use another GPU. The existing feature grid for audit
subjects is reused; the first E14 run still has to encode the new CAR-only
probe-train and probe-validation splits. It then trains two q4 probes, runs
strict E12 for each seed on CPU, and performs a crossed hierarchical bootstrap
over probe seed, batch repeat, and held-reference identity. Seed 7 is excluded
from all confirmatory statistics.

Push these complete directories:

```text
outputs/reve_set_probe_confirmation_logits_q4_s21
outputs/reve_set_probe_confirmation_logits_q4_s42
outputs/reve_set_class_safeguard_audit_s21
outputs/reve_set_class_safeguard_audit_s42
outputs/reve_set_probe_seed_confirmation
```

E14 disables probe checkpoint saving, so the logit directories contain no
large `.pt` files. The main decision file is
`probe_seed_confirmation_summary.json`. A positive result requires every new
seed to improve mean RMSE against both strict E11 and topology in all three
frozen regimes, hierarchical RMSE intervals below zero, task-metric
noninferiority, and no material mean harm. Class-wise results remain a
falsification diagnostic and cannot revive the current class-uniform claim.
Even a pass advances the frozen mean method only to an external open EEG
dataset; two untouched seeds on one dataset are not a paper-level claim.

## Training-time operator-consistency screen

E14 failed its untouched-seed gate: seed 21 did not beat both strong baselines
in every frozen regime, and the hierarchical candidate-versus-topology RMSE
intervals crossed zero. This closes the current post-hoc class-cap method. E15
does not tune another cap on subjects 71--89. It instead asks a new source-only
representation/readout question: can the q4 REVE set probe learn the nested
64/32/16-channel observation operators during training?

E15 trains three strict controls with probe seed 7:

- CAR-only supervised CE;
- CAR/native32/native16 supervised multi-view CE; and
- the same multi-view CE plus CAR-teacher operator consistency. The full-CAR
  prediction is detached and supplies KL targets to native32 and native16 with
  frozen weights 0.5 and 1.0.

This third arm must beat the second arm, so ordinary montage augmentation
cannot be relabeled as a rule-informed contribution. All arms train on subjects
1--60, early-stop on 61--70, and are screened on development subjects 71--89.
Reserved test subjects 90--109 remain unencoded and unscored.

The completed decision file is
`outputs/reve_set_operator_consistency_screen_s7/operator_consistency_summary.json`.
E15 failed its predeclared gate. On `native16@CAR`, CAR-only, multi-view CE and
operator consistency obtained BAcc 0.3470, 0.4561 and 0.4546 respectively.
The rule arm improved over CAR-only but did not beat ordinary multi-view CE and
missed clean-CAR noninferiority. The justified decision is therefore to retain
multi-view CE as a baseline and drop the unsupported rule-loss claim.

## Locked development baseline sweep

E15 showed that ordinary multi-view CE, rather than its CAR-teacher KL term,
explained the native-montage recovery. Before another custom method is designed,
run the frozen-REVE baseline matrix across reference augmentation, structured
and random native montage removal, motor-region dropout, joint augmentation,
and a generic Jensen--Shannon consistency control:

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make benchmark-baselines
```

The sweep uses probe seeds 7/21/42 and scores only development subjects 71--89.
PhysioNetMI subjects 90--109 were inspected in earlier E3--E8 work and are no
longer described as a globally untouched paper test. A paper-level method must
be confirmed on an external dataset. See
[docs/BASELINE_PLAN.md](docs/BASELINE_PLAN.md) for the literature rationale,
exact run matrix, primary metric, and required second-stage external baselines.

## Phase-B channel-adaptation screen

The next locked screen tests an existing spherical-spline interpolation
adapter against the Phase-A `car_only` and `joint_multiview_ce` controls. REVE
remains frozen, and the adapter is explicitly a literature baseline rather
than a GaugeEEG contribution:

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make channel-adaptation
```

See [docs/PHASE_B_CHANNEL_ADAPTATION.md](docs/PHASE_B_CHANNEL_ADAPTATION.md)
for the exact matrix, primary-metric bootstrap, and external-confirmation lock.

## Gauge-Quotient Bilateral Adapter screen

Phase B did not establish spherical-spline interpolation as an improvement
over the native flexible-channel baseline. The first custom representation
screen therefore keeps frozen REVE and the locked joint multi-view CE training,
then adds a matched-capacity bilateral spectral branch. Its odd and even
spatial contrasts are exactly invariant to a common EEG reference while
separately preserving lateralized and bilateral motor activity:

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make gqba-screen
```

The runner evaluates a raw spectral capacity control, odd-only ablation, and
the complete odd+even GQBA over seeds 7/21/42. See
[docs/GQBA_SCREEN.md](docs/GQBA_SCREEN.md) for the fixed representation,
matched-parameter checks, advancement gates, and result files to return.

# Research plan

## 1. Starting point

NAI-SSL argues that a self-supervised objective can be stronger when its
supervision is derived from a scientific property of the data rather than
copied from natural-image learning. Its specific rule is inter-hemispheric
structural covariance in MRI. GaugeEEG keeps the design principle but changes
the modality, rule, benchmark, and anchor baseline.

The high-tier anchor is REVE (NeurIPS 2025), not NAI-SSL. REVE is a strong and
appropriate target because it explicitly addresses heterogeneous electrode
layouts, provides code and pretrained weights, and reports results on
PhysioNetMI. This lets us ask a narrower question that its spatial positional
encoding does not explicitly encode: is the representation invariant to the
voltage reference convention?

## 2. Scientific rule

Scalp voltage has no physically meaningful absolute zero. If an EEG sample is
represented by `X` with `C` channels, then

```text
X' = X + 1 a(t)
```

describes the same pairwise voltage field for any common time-varying signal
`a(t)`. A linear reference with weights `w`, where `w^T 1 = 1`, is

```text
R_w(X) = X - 1 (w^T X).
```

Pairwise channel differences are invariant to this transformation. Common
average referencing is a canonical projection onto the subspace orthogonal to
the all-ones channel vector.

This is a *gauge symmetry*: multiple numerical arrays represent the same
underlying measurable voltage differences.

## 3. Candidate limitation

REVE's 4D positional encoding models where an electrode is and when a patch
occurs. The released PhysioNetMI preprocessing fixes average reference before
training/evaluation. The architecture and MAE loss do not explicitly identify
signals that differ only by a reference operator as equivalent.

Therefore, arbitrary-layout support does not automatically imply
reference-convention invariance. This is a testable limitation, not an assumed
fact; Experiment E2 is designed to falsify it.

## 4. Research gap

The proposed gap is a controlled evaluation and learning framework for EEG
foundation-model robustness to valid reference transformations while holding
the subject, task, electrode locations, and underlying recording fixed.

The defensible claim at pilot stage is:

> Existing EEG representation benchmarks mainly vary subjects, datasets, and
> electrode layouts, but do not isolate reference convention as a structured
> nuisance variable in frozen-representation transfer.

A final paper must repeat a systematic literature search before claiming that
no prior study has addressed this problem.

## 5. Contribution ladder

1. **RefShift-EEG benchmark.** Paired, label-preserving reference views with
   task performance and representation-drift metrics.
2. **Exact sanity baseline.** CAR canonicalization proves the benchmark is
   measuring the intended additive gauge component.
3. **Prior-confounding benchmark and class-safe operator correction.** Expose
   the identifiability boundary between reference bias and unknown label shift,
   then regularize uncertain target corrections with the observation operator
   and an explicit worst-class safeguard.
4. **Beyond ideal common-mode shifts.** Extend to bipolar derivations,
   missing-channel montages, and cross-dataset transfer, where exact CAR
   canonicalization alone is insufficient.

Only item 1 and the exact baseline are implemented in v0.1. The learned method
should be added only if the pilot establishes a meaningful failure mode.

Current evidence has advanced beyond v0.1: a validation-selected prediction
consistency loss improves held-out full-montage Cz robustness over ordinary
multi-view augmentation. This remains a frozen-encoder robust readout. E7b's
native construction was algebraically valid, but released attention pooling
plus a linear probe failed the clean-performance gate. E7c therefore tests a
variable-set frozen-token readout before attributing any error to the montage.

The first E7a missing-channel implementation zero-filled absent electrodes and
caused all probes to collapse near four-class chance. Because REVE accepts
channel subsets and their positions natively, that result is an input-modeling
artifact rather than evidence of an inherent foundation-model limitation. E7b
therefore validates native removal with a fixed-dimensional attention-pooled
readout before any montage-aware method is trained. E7c uses validation-only
selection of a pooling-by-multihead-attention query count, followed by the same
clean and noncollapse gates. Only a passing E7c result licenses montage-aware
training and repeated-seed method comparisons.

E7c subsequently passed: the q4 set readout reached 0.6112 clean CAR BAcc and
0.3850 on native16@Cz, with a 0.2263 paired subject-level gap. The input no
longer exhibits E7a's single-class collapse, but post-hoc inspection shows a
functional two-class failure: both-fists and both-feet recall approach zero
while their one-vs-rest AUROC remains above chance. E7d closes the missing
full-reference comparison and quantifies this recoverable class-conditional
decision bias before the learned montage method is introduced.

E7d also showed that the near-zero native16 CAR-versus-Cz aggregate gap does
not establish native gauge invariance: those predictions still disagreed on
11.5% of trials, and Cz lies near the center of the selected motor montage.
E7e therefore evaluates predeclared Pz/Fz alternatives at native16/native32.
Its suite-level criterion decides whether the proposed method should jointly
target gauge and montage transformations or treat gauge consistency as an
auxiliary to a montage-primary method.

E8 subsequently localized the dominant recoverable error to the readout. A
three-parameter target-view class bias reduced the worst within-montage recall
gap by 67.3% and improved worst native BAcc, while leave-one-view-out pooled
bias over-corrected Pz and increased the worst recall gap. The current research
gap is therefore not generic representation invariance: it is predicting a
reference- and montage-conditional correction without labeled trials from the
target observation operator.

E9 showed that full-batch logit statistics predict the oracle bias almost
perfectly, but subsequent analysis exposed a stronger identity: bias-only NLL
depends on labels only through their class prior. Because the motor-imagery cue
schedule is nearly balanced, a known uniform prior recovers the supervised
oracle without learned manifold regression. The remaining defensible gap is
small-batch and prior-robust adaptation. E10 therefore tests whether topology
provides a useful inductive bias when unlabeled batches are scarce, and
explicitly falsifies the method under unknown or shifted class proportions.

E10 confirmed both sides of that claim. Topology shrinkage substantially
improved random batches of 32 trials, but severe-skew error was three times the
balanced-batch error. Exploratory E11 then tested regularized soft-confusion
inversion with disjoint prior-model and adaptation subjects. It significantly
reduced mean severe-skew bias RMSE, yet one of four dominant-class directions
worsened slightly. It also could not recover the true severe batch prior
accurately. Post-run review found that topology fitting still overlapped the
adaptation subjects, so E11's archived numbers are diagnostic rather than
strict cross-subject evidence.

The resulting empirical gap is narrower and more defensible: frozen logits can
support a useful *partial* correction under unknown prior shift, but average
improvement does not provide class-uniform safety. The next proposed method
must detect or constrain harmful class-specific updates without target labels,
while retaining E10's small-batch benefit. E11 is evidence for this problem
definition, not yet evidence that the proposed correction is publishable.

E12 implements the resulting method hypothesis. A leave-one-reference-out
source procedure learns class-wise upper bounds on E11's update in full
zero-sum bias space. Source, adaptation, and evaluation subjects are now fully
disjoint. The pre-push strict screen reduced severe RMSE by 42.3%, beat
topology-only, and improved all four class directions by point estimate. The
right-fist clustered interval still crossed zero, so E12 is a promising
single-seed screen that must be repeated and transferred to an external
dataset before supporting a class-uniform paper claim.

E13 then exposed a stronger limitation in that interpretation: improvement
against E10's fixed shrinkage is not enough when either strict E11 or static
topology is better for a particular regime or class. A post-hoc paired audit
found that E12 improves mean RMSE against both strong baselines across random,
balanced, and severe conditions, although the primary interval versus strict
E11 still crosses zero. At class level, right-fist-dominant E12 is
significantly worse than strict E11. The defensible next experiment is thus a
new-seed confirmation of the mean-only claim. The present class-wise cap cannot
support a class-uniform novelty claim without redesign on source data followed
by validation on untouched external data.

Before that confirmation, E14 protocol review found that the old q4 probe was
early-stopped using subjects 71--89, the same pool later divided by E12 into
source, adaptation, and evaluation subjects. Thus E12/E13 remain valid paired
diagnostics among methods sharing one frozen probe, but seed 7 cannot count as
independent confirmation. E14 repairs this with probe train 1--60, probe
validation 61--70, downstream audit 71--89, and reserved test 90--109. Its
confirmatory analysis uses only unseen probe seeds 21 and 42 under the frozen
E13 mean-only rule.

## 6. Core hypotheses

- **H1:** Frozen REVE embeddings and linear-probe performance change under
  valid single-electrode or linear re-referencing.
- **H2:** Exact CAR canonicalization removes the simple reference component and
  recovers the clean-view representation, validating the benchmark.
- **H3:** Under harder convention changes where exact canonicalization is
  unavailable, gauge-consistency learning improves worst-reference performance
  without reducing clean-reference performance.
- **H4:** Under unknown target class proportions, an operator-conditioned
  correction can improve mean severe-shift error while preserving random and
  balanced conditions. E13 makes this the current testable claim; improvement
  for every dominant class is a separate, currently falsified hypothesis
  because E12 harms right-fist performance relative to strict E11.

## 7. Go/no-go criteria

Proceed to a learned method if, across repeated seeds and the full subject
split:

- at least one valid reference produces a >=3 percentage-point balanced
  accuracy drop in the unprotected frozen model;
- the direction of the drop is stable across seeds;
- embedding drift is nontrivial (for example, mean paired cosine <0.95 or a
  clear CKA reduction); and
- CAR canonicalization recovers at least 80% of the simple-reference gap.

Pivot to bipolar/missing-channel transfer if all simple-reference drops are
<1 point and paired cosine is >0.98. Stop this direction if the harder shifts
also produce no meaningful drift or downstream loss.

## 8. Threats to validity

- PhysioNetMI is one task and one acquisition family; a paper needs at least one
  additional open dataset.
- REVE weights require acceptance of a responsible-use agreement, so the
  bandpower experiment is the fully ungated smoke test.
- A single reference channel may contain sensor noise; report both deterministic
  named references and a distributed linear reference.
- CAR is an exact solution only when all channels are retained and reference
  changes are purely common-mode.
- Results from the 12-subject pilot are for debugging and effect-size screening,
  not paper claims.
- E12 is still one probe seed and one dataset. Its source-only caps are fixed
  before target audit, but external transfer and repeated-seed uncertainty are
  mandatory before a method claim.
- E13's strongest-baseline rule was defined after inspecting seed 7. It may
  eliminate unsupported claims, but only new probe seeds and an untouched
  external dataset can provide confirmatory evidence.
- E12/E13 seed 7 shares its probe early-stopping pool with the downstream audit
  pool. It is retained for hypothesis formation and paired failure analysis,
  but excluded from E14 confirmation. E14's four-way split prevents this
  leakage for new seeds.

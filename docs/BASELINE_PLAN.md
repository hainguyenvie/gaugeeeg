# GaugeEEG benchmark lock and baseline plan

Literature search date: 2026-07-20.

## Scope and current evidence

GaugeEEG isolates two related but distinct shifts:

1. **Reference/gauge shift:** the same signals and electrode set are expressed
   under another physically valid voltage reference.
2. **Observation/montage shift:** only a subset of electrodes remains observed;
   the reference is then defined within that retained montage.

The benchmark must report the two main effects and their interaction rather
than collapse them into one robustness number. The locked development grid is:

```text
full / native32 / native16  x  CAR / Cz / Pz / Fz
```

PhysioNetMI subjects 90--109 have already been inspected during E3--E8. The
repository therefore treats all PhysioNetMI results as development evidence.
An external dataset, not another relabeling of these subjects, is required for
paper-level confirmation.

## What the literature does and does not cover

The search found substantial adjacent work, but no paper using GaugeEEG's
exact paired design: one recording, label and electrode field held fixed while
only a lawful voltage-reference representative changes.

- [REVE](https://brain-bzh.github.io/reve/) accepts arbitrary electrode
  coordinates and reports transfer across setups. It does not explicitly
  identify reference-equivalent arrays in its objective.
- [Learning Topology-Agnostic EEG Representations with Geometry-Aware
  Modeling](https://proceedings.neurips.cc/paper_files/paper/2023/file/a8c893712cb7858e49631fb03c941f8d-Paper-Conference.pdf)
  maps heterogeneous montages to a unified topology. It is a close model-level
  comparator for montage transfer, not a direct gauge-invariance result.
- [Robust learning from corrupted EEG with dynamic spatial
  filtering](https://arxiv.org/abs/2105.12916) targets missing/noisy channels
  and motivates random/region corruption baselines.
- [Channel Adaptation for EEG Foundation
  Models](https://arxiv.org/abs/2604.23091) compares Conv1d projection,
  spherical-spline interpolation, source-space decomposition and Riemannian
  re-centering across foundation models. These are required external-adaptation
  comparators for the montage axis.
- [Beyond Accuracy: Robustness, Interpretability and Expressiveness of EEG
  Foundation Models](https://arxiv.org/abs/2605.17562) evaluates random and
  region channel dropout and reports that removal and zero padding can lead to
  materially different conclusions. This supports GaugeEEG's native-removal
  correction after E7a.
- [LUNA](https://proceedings.neurips.cc/paper_files/paper/2025/file/66969a9e6bd7a26dfeccea7227178ca7-Paper-Conference.pdf)
  and [DIVER-0](https://arxiv.org/abs/2507.14141) are architecture-level
  topology/channel comparators. Their published numbers cannot be compared
  directly with GaugeEEG because the datasets, tasks and protocols differ;
  they must be run inside the same benchmark.
- Classical re-reference estimation, including [robust reference
  estimation](https://pubmed.ncbi.nlm.nih.gov/24975291/), addresses how to
  choose a reference. GaugeEEG instead asks whether a learned representation is
  stable across reference-equivalent inputs.

The novelty claim must therefore remain: **a paired gauge-reference benchmark
and, only if supported, a rule-informed method evaluated against the closest
montage/channel baselines.** It must not claim that montage robustness or EEG
invariance in general is new.

## Phase A: reproducible frozen-REVE baseline lock

Run with:

```bash
git pull
source .venv/bin/activate
make test
DEVICE=cuda make benchmark-baselines
```

Set `DEVICE=cuda:1` or `SEEDS="7 21 42"` as needed. The runner uses subjects
1--60 for probe training, 61--70 for early stopping and 71--89 for development
evaluation. It never scores 90--109.

The complete matrix uses probe seeds 7, 21 and 42:

| Baseline | Training views | Purpose |
|---|---|---|
| `car_only` | full CAR | Clean frozen-representation control |
| `reference_multiview_ce` | full CAR/Pz/Fz | Ordinary reference augmentation; Cz held out |
| `structured_montage_ce` | full/native32/native16 under CAR | Motor-centric montage augmentation |
| `joint_multiview_ce` | full/native32/native16 under CAR/Pz/Fz | Strong joint augmentation; Cz held out |
| `random_montage_ce` | three deterministic nested random 32/16 layouts | Random channel-removal augmentation |
| `region_dropout_ce` | left/right motor-region removal | Structured region-corruption augmentation |
| `joint_js_consistency` | same views as joint CE plus generalized JS | Generic consistency control, not a GaugeEEG novelty claim |

The analyzer validates the exact splits, views, objectives, dataset fingerprint,
resolved REVE/position-model revisions and absence of PhysioNet test scoring. It
writes:

```text
outputs/reve_benchmark_lock/aggregate/baseline_manifest.csv
outputs/reve_benchmark_lock/aggregate/baseline_metrics_by_seed.csv
outputs/reve_benchmark_lock/aggregate/baseline_metrics_summary.csv
outputs/reve_benchmark_lock/aggregate/baseline_method_summary.csv
outputs/reve_benchmark_lock/aggregate/baseline_pairwise_bootstrap.csv
outputs/reve_benchmark_lock/aggregate/benchmark_lock_summary.json
```

Baseline selection is development-only. A method is eligible only when its
clean CAR BAcc is within 0.01 of CAR-only. The primary ranking metric is mean
BAcc over `native16@{CAR,Cz,Pz,Fz}`. The analyzer also reports clean, full
reference, native32, suite-worst and worst-class metrics plus a hierarchical
probe-seed-by-subject bootstrap.

## Phase B: literature implementations required before a final paper claim

Phase A determines the strongest controlled baseline already supported by the
repository. It is not the final literature comparison. Add, in this order:

1. Spherical-spline interpolation to the full channel set.
2. A learned Conv1d/channel projection baseline.
3. Riemannian re-centering or the closest reproducible channel-adaptation
   method from Kokate et al.
4. Dynamic Spatial Filtering or an equivalently faithful missing-channel
   implementation.
5. At least one topology-agnostic/flexible EEG model besides REVE, subject to
   released code, weights and compatible licensing.

Every imported method must use the same subject split, input epochs, evaluation
views and metrics. Published table values from another dataset are context,
not leaderboard entries.

## Phase C: matched trainable-encoder controls

Only after the frozen benchmark is locked, compare:

```text
frozen encoder       : Multi-view CE  vs  Multi-view CE + proposed rule
adapter/partial tune : Multi-view CE  vs  Multi-view CE + proposed rule
full fine-tune       : Multi-view CE  vs  Multi-view CE + proposed rule
```

The architecture, trainable parameter budget, optimization schedule and input
views must be matched within each row. A full-unfreeze rule method cannot be
credited for beating a frozen baseline.

## Paper-level advancement rule

A proposed method advances only if it:

1. beats the strongest locked baseline on the predeclared primary metric with
   a paired interval above zero;
2. preserves clean CAR within 0.01;
3. preserves native32 and worst-class recall;
4. improves consistently across new optimization seeds;
5. survives ablation against the identical training pipeline without the
   scientific rule; and
6. reproduces on an external open EEG dataset that was not used to design the
   benchmark or method.

Passing Phase A only selects a development baseline. It is not a paper-level
method result.

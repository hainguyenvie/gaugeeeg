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

After the clean gate passes, the existing full configuration can still be used
for the lightweight attention-pooled ablation. A token-probe stress config will
be enabled after the clean run establishes which released LP variant is
reproducible on the local environment.

```bash
gaugeeeg run \
  --config configs/full_physionetmi.yaml \
  --encoder reve \
  --device cuda \
  --output-dir outputs/full_reve
```

The full experiment downloads six motor-imagery runs for all 109 subjects and
can require several GB of disk and substantial feature-extraction time.

## Reading the output

Each run creates:

- `metrics.csv`: one row per defense and test reference.
- `summary.json`: compact best/worst gaps and run metadata.
- `resolved_config.yaml`: the exact configuration used.
- `feature_cache/`: reusable frozen features for interrupted or repeated runs.

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

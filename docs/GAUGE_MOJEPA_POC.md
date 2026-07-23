# Gauge-MOJEPA LoRA proof of concept

## Why this screen exists

Every GaugeEEG experiment through GQRA extracts REVE tokens under
`torch.inference_mode()` and optimizes only a downstream head. Those negative
results therefore falsify the tested frozen-readout methods, but they do not
test whether a small foundation-model update can repair the representation.

This screen makes that distinction explicit. It adapts only low-rank attention
updates and then discards the pretext head. The adapted encoder is evaluated by
the same q4 variable-set probe, subject splits, observation views and audit
subjects used by the locked Phase-A benchmark.

## Proposed method

For an underlying potential field `X`, montage selection `S_M` and reference
weights `a`, the observation is

```text
Y_(M,a) = S_M (X - 1 a^T X) + noise.
```

A reference change at fixed `M` is a gauge nuisance: pairwise channel
differences retain the same information. Reducing `M` is different because it
removes sensors and is lossy. Gauge-MOJEPA therefore uses two different rules:

1. exact representation consistency only between references of the same
   montage;
2. asymmetric prediction from every lossy student view to a frozen full-CAR
   teacher, conditioned on reference identity, acquisition policy, retained
   fraction and the exact binary sensor mask in canonical channel order.

The train-time objective is

```text
L = L_predict + 0.25 L_same-montage-gauge + 0.10 L_full-CAR-anchor.
```

`L_predict` is cosine latent prediction. `L_same-montage-gauge` never compares
different channel sets as if they were information-equivalent. The anchor
limits drift on clean full-CAR. The teacher is the released REVE encoder with
the adapter disabled; the student is the same encoder with LoRA enabled.

LoRA rank 8 is attached to QKV and output projections of the final four
attention blocks. Both REVE's released `to_qkv`/`to_out` layout and PyTorch
fused `MultiheadAttention` are supported. Only the adapter tensors are written
to `adapter.pt`; REVE weights are not copied into the checkpoint.

## Matched arms

| Arm | Labels in adaptation | Teacher prediction | Operator code | Gauge + anchor |
|---|---:|---:|---:|---:|
| `joint_multiview_ce` | downstream only | no | no | no |
| `lora_multiview_ce` | yes | no | no | no |
| `lora_generic_jepa` | no | yes | zeroed | no |
| `gauge_mojepa` | no | yes | yes | yes |

The first arm is the existing frozen-REVE control. The other three use exactly
the same adapter rank, attention targets and adaptation views. This separates:

- backbone adaptation from a frozen readout;
- generic predictive adaptation from measurement-operator conditioning;
- the proposed gauge rule from matched LoRA capacity.

## Leakage lock

- Subjects 1--60: adaptation and downstream probe training.
- Subjects 61--70: adaptation early stopping and probe early stopping.
- Subjects 71--89: development audit only.
- Subjects 90--109: neither adaptation nor scoring. They have been inspected
  by older experiments and are not called a globally untouched paper test.
- A passing development result still requires confirmation on an external EEG
  dataset.

## Stage-0 run

First accept access to `brain-bzh/reve-base`, authenticate with Hugging Face,
and make sure the locked `joint_multiview_ce_s7` control exists. Then run:

```bash
git pull
source .venv/bin/activate
pip install -e ".[data,reve,dev]"
make test
DEVICE=cuda SEEDS="7" make gauge-mojepa-poc
```

The checked-in Stage-0 configuration deliberately caps adaptation to 256 train
trials and 64 validation trials for two epochs. Its purpose is to establish
that the gated model loads, the expected final four attention blocks receive
LoRA, gradients update the adapter, checkpoints reload, and all three arms
reach the existing audit pipeline. It is not a paper result.

Expected files for every new arm are:

```text
outputs/reve_gauge_mojepa_poc/adapt/<method>_s7/adapter.pt
outputs/reve_gauge_mojepa_poc/adapt/<method>_s7/adaptation_summary.json
outputs/reve_gauge_mojepa_poc/eval/<method>_s7/summary.json
outputs/reve_gauge_mojepa_poc/eval/<method>_s7/validation_predictions.csv
```

The adaptation summary records the resolved REVE revisions, exact target
modules, trainable adapter count, checkpoint hash, per-epoch loss components,
label usage and confirmation that reserved test subjects were not used.

## Full development screen

Only after Stage 0 completes, copy the configuration and change
`max_train_trials` and `max_val_trials` to `null`. Increase the epoch budget
based on compute while keeping the loss weights and views fixed, then run all
three seeds:

```bash
CONFIG=configs/reve_gauge_mojepa_full.yaml \
DEVICE=cuda SEEDS="7 21 42" make gauge-mojepa-poc
```

Do not tune weights on subjects 71--89. Select epochs using only the adaptation
loss on subjects 61--70 and keep the downstream probe protocol identical for
all arms.

## Advancement rule

Gauge-MOJEPA is a viable method contribution only if, over seeds 7/21/42:

1. native16 balanced accuracy beats `joint_multiview_ce`,
   `lora_multiview_ce` and `lora_generic_jepa` with a positive hierarchical
   bootstrap interval;
2. clean-CAR and native32 BAcc are non-inferior within 0.01;
3. no native16 class recall loses more than 0.01 against the strongest matched
   control;
4. the adapter layout and parameter count are identical across all LoRA arms;
5. the result is reproduced on a predeclared external dataset.

If the method only beats frozen REVE, the result supports LoRA adaptation but
not the Gauge-MOJEPA rule. If it fails to beat generic JEPA, the
measurement-operator contribution is unsupported.

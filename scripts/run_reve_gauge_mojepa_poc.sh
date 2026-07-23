#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_gauge_mojepa_poc.yaml}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-7}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/reve_benchmark_lock}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reve_gauge_mojepa_poc}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

METHODS=(lora_multiview_ce lora_generic_jepa gauge_mojepa)
read -r -a SEED_ARRAY <<< "${SEEDS}"

for seed in "${SEED_ARRAY[@]}"; do
  baseline="${BASELINE_ROOT}/joint_multiview_ce_s${seed}"
  if [[ ! -f "${baseline}/summary.json" ]]; then
    echo "Missing locked control: ${baseline}" >&2
    echo "Run DEVICE=${DEVICE} SEEDS='${seed}' make benchmark-baselines first." >&2
    exit 1
  fi

  for method in "${METHODS[@]}"; do
    adaptation_dir="${OUTPUT_ROOT}/adapt/${method}_s${seed}"
    checkpoint="${adaptation_dir}/adapter.pt"
    evaluation_dir="${OUTPUT_ROOT}/eval/${method}_s${seed}"
    if [[ ! -f "${checkpoint}" ]]; then
      "${GAUGEEG[@]}" adapt-mojepa \
        --config "${CONFIG}" \
        --objective "${method}" \
        --device "${DEVICE}" \
        --seed "${seed}" \
        --output-dir "${adaptation_dir}"
    else
      echo "Reusing ${checkpoint}"
    fi
    if [[ ! -f "${evaluation_dir}/summary.json" \
        || ! -f "${evaluation_dir}/validation_predictions.csv" ]]; then
      "${GAUGEEG[@]}" run \
        --config "${CONFIG}" \
        --device "${DEVICE}" \
        --probe-seed "${seed}" \
        --reference-seed 7 \
        --adapter-checkpoint "${checkpoint}" \
        --output-dir "${evaluation_dir}"
    else
      echo "Reusing ${evaluation_dir}"
    fi
  done
done

echo "Gauge-MOJEPA Stage-0 runs completed under ${OUTPUT_ROOT}."
echo "Return adaptation_summary.json, summary.json, and validation_predictions.csv for each arm."

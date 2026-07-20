#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_benchmark_lock.yaml}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-7 21 42}"
CONTROL_ROOT="${CONTROL_ROOT:-outputs/reve_benchmark_lock}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reve_channel_adaptation}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

run_ssi_arm() {
  local method="$1"
  local seed="$2"
  local objective="$3"
  shift 3
  local training_views=("$@")
  local output_dir="${OUTPUT_ROOT}/${method}_s${seed}"

  if [[ -f "${output_dir}/summary.json" \
      && -f "${output_dir}/validation_predictions.csv" ]]; then
    echo "Reusing ${method} seed ${seed}: ${output_dir}"
  else
    echo "Running ${method} seed ${seed}"
    "${GAUGEEG[@]}" run \
      --config "${CONFIG}" \
      --device "${DEVICE}" \
      --probe-seed "${seed}" \
      --reference-seed 7 \
      --set-queries 4 \
      --probe-objective "${objective}" \
      --consistency-weight 0.0 \
      --defenses spherical_spline \
      --training-views "${training_views[@]}" \
      --output-dir "${output_dir}"
  fi
  RUN_SPECS+=("${method}=${output_dir}")
}

RUN_SPECS=()
read -r -a SEED_ARRAY <<< "${SEEDS}"
first_seed="${SEED_ARRAY[0]}"
predictions_archive="${CONTROL_ROOT}/validation_predictions.tar.gz"
if [[ ! -f "${CONTROL_ROOT}/car_only_s${first_seed}/validation_predictions.csv" \
    && -f "${predictions_archive}" ]]; then
  echo "Extracting committed Phase-A predictions archive"
  tar -xzf "${predictions_archive}" \
    -C "${CONTROL_ROOT}" \
    --strip-components=2 \
    --no-same-owner \
    --skip-old-files
fi
for seed in "${SEED_ARRAY[@]}"; do
  car_control="${CONTROL_ROOT}/car_only_s${seed}"
  joint_control="${CONTROL_ROOT}/joint_multiview_ce_s${seed}"
  if [[ ! -f "${car_control}/summary.json" \
      || ! -f "${car_control}/validation_predictions.csv" \
      || ! -f "${joint_control}/summary.json" \
      || ! -f "${joint_control}/validation_predictions.csv" ]]; then
    echo "Missing Phase-A controls for seed ${seed} under ${CONTROL_ROOT}." >&2
    echo "Run 'make benchmark-baselines' or set CONTROL_ROOT first." >&2
    exit 1
  fi
  RUN_SPECS+=("car_only=${car_control}")
  RUN_SPECS+=("joint_multiview_ce=${joint_control}")
  run_ssi_arm ssi_car_only "${seed}" car_only car
  run_ssi_arm ssi_joint_multiview_ce "${seed}" multi_view_ce \
    car pz fz \
    "native32@car" "native32@pz" "native32@fz" \
    "native16@car" "native16@pz" "native16@fz"
done

"${GAUGEEG[@]}" channel-adaptation-audit \
  --runs "${RUN_SPECS[@]}" \
  --expected-seeds "${SEED_ARRAY[@]}" \
  --output-dir "${OUTPUT_ROOT}/aggregate"

echo "Completed the Phase-B spherical-spline channel-adaptation screen."
python -m json.tool "${OUTPUT_ROOT}/aggregate/channel_adaptation_summary.json"

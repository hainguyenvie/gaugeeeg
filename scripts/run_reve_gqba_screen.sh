#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_benchmark_lock.yaml}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-7 21 42}"
CONTROL_ROOT="${CONTROL_ROOT:-outputs/reve_benchmark_lock}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reve_gqba_screen}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

TRAINING_VIEWS=(
  car pz fz
  "native32@car" "native32@pz" "native32@fz"
  "native16@car" "native16@pz" "native16@fz"
)

run_candidate() {
  local method="$1"
  local seed="$2"
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
      --probe-objective multi_view_ce \
      --consistency-weight 0.0 \
      --probe-auxiliary "${method}" \
      --training-views "${TRAINING_VIEWS[@]}" \
      --output-dir "${output_dir}"
  fi
  RUN_SPECS+=("${method}=${output_dir}")
}

RUN_SPECS=()
read -r -a SEED_ARRAY <<< "${SEEDS}"
for seed in "${SEED_ARRAY[@]}"; do
  car_control="${CONTROL_ROOT}/car_only_s${seed}"
  joint_control="${CONTROL_ROOT}/joint_multiview_ce_s${seed}"
  for control in "${car_control}" "${joint_control}"; do
    if [[ ! -f "${control}/summary.json" \
        || ! -f "${control}/validation_predictions.csv" ]]; then
      echo "Missing Phase-A control: ${control}" >&2
      echo "Run 'make benchmark-baselines' or set CONTROL_ROOT first." >&2
      exit 1
    fi
  done
  RUN_SPECS+=("car_only=${car_control}")
  RUN_SPECS+=("joint_multiview_ce=${joint_control}")
  run_candidate spectral_capacity_control "${seed}"
  run_candidate gqba_odd "${seed}"
  run_candidate gqba_odd_even "${seed}"
done

"${GAUGEEG[@]}" gqba-audit \
  --runs "${RUN_SPECS[@]}" \
  --expected-seeds "${SEED_ARRAY[@]}" \
  --output-dir "${OUTPUT_ROOT}/aggregate"

echo "Completed the matched-capacity GQBA development screen."
python -m json.tool "${OUTPUT_ROOT}/aggregate/gqba_summary.json"

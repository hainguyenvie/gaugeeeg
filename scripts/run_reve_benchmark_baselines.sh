#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_benchmark_lock.yaml}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-7 21 42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reve_benchmark_lock}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

run_arm() {
  local method="$1"
  local seed="$2"
  local objective="$3"
  local weight="$4"
  shift 4
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
      --consistency-weight "${weight}" \
      --training-views "${training_views[@]}" \
      --output-dir "${output_dir}"
  fi
  RUN_SPECS+=("${method}=${output_dir}")
}

RUN_SPECS=()
read -r -a SEED_ARRAY <<< "${SEEDS}"
for seed in "${SEED_ARRAY[@]}"; do
  run_arm car_only "${seed}" car_only 0.0 car
  run_arm reference_multiview_ce "${seed}" multi_view_ce 0.0 car pz fz
  run_arm structured_montage_ce "${seed}" multi_view_ce 0.0 \
    car "native32@car" "native16@car"
  run_arm joint_multiview_ce "${seed}" multi_view_ce 0.0 \
    car pz fz \
    "native32@car" "native32@pz" "native32@fz" \
    "native16@car" "native16@pz" "native16@fz"
  run_arm random_montage_ce "${seed}" multi_view_ce 0.0 \
    car \
    "native_random32_s101@car" "native_random16_s101@car" \
    "native_random32_s202@car" "native_random16_s202@car" \
    "native_random32_s303@car" "native_random16_s303@car"
  run_arm region_dropout_ce "${seed}" multi_view_ce 0.0 \
    car "native_drop_left_motor@car" "native_drop_right_motor@car"
  run_arm joint_js_consistency "${seed}" rule_consistency 1.0 \
    car pz fz \
    "native32@car" "native32@pz" "native32@fz" \
    "native16@car" "native16@pz" "native16@fz"
done

"${GAUGEEG[@]}" baseline-benchmark-audit \
  --runs "${RUN_SPECS[@]}" \
  --expected-seeds "${SEED_ARRAY[@]}" \
  --output-dir "${OUTPUT_ROOT}/aggregate"

echo "Completed the frozen-REVE development baseline sweep."
echo "PhysioNetMI subjects 90--109 were not scored by these runs, but are historically inspected."
python -m json.tool "${OUTPUT_ROOT}/aggregate/benchmark_lock_summary.json"

#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_benchmark_lock.yaml}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-7 21 42}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/reve_benchmark_lock}"
GQBA_ROOT="${GQBA_ROOT:-outputs/reve_gqba_screen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reve_gsra_screen}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

TRAINING_VIEWS=(
  car pz fz
  "native32@car" "native32@pz" "native32@fz"
  "native16@car" "native16@pz" "native16@fz"
)

restore_prediction() {
  local run_dir="$1"
  local archive="$2"
  local prediction="${run_dir}/validation_predictions.csv"
  if [[ ! -f "${prediction}" && -f "${archive}" ]]; then
    echo "Restoring ${prediction} from ${archive}"
    tar --no-same-owner -xzf "${archive}" "${prediction}"
  fi
}

run_candidate() {
  local method="$1"
  local auxiliary="$2"
  local preservation_weight="$3"
  local residual_consistency_weight="$4"
  local gate_supervision_weight="$5"
  local seed="$6"
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
      --probe-auxiliary "${auxiliary}" \
      --auxiliary-fusion gated_residual \
      --auxiliary-gate-initial-probability 0.25 \
      --auxiliary-preservation-weight "${preservation_weight}" \
      --auxiliary-residual-consistency-weight "${residual_consistency_weight}" \
      --auxiliary-gate-supervision-weight "${gate_supervision_weight}" \
      --auxiliary-target-classes 2 3 \
      --training-views "${TRAINING_VIEWS[@]}" \
      --output-dir "${output_dir}"
  fi
  RUN_SPECS+=("${method}=${output_dir}")
  PREDICTION_FILES+=("${output_dir}/validation_predictions.csv")
}

RUN_SPECS=()
PREDICTION_FILES=()
read -r -a SEED_ARRAY <<< "${SEEDS}"
for seed in "${SEED_ARRAY[@]}"; do
  joint_control="${BASELINE_ROOT}/joint_multiview_ce_s${seed}"
  prior_gqba="${GQBA_ROOT}/gqba_odd_even_s${seed}"
  restore_prediction "${joint_control}" "${BASELINE_ROOT}/validation_predictions.tar.gz"
  restore_prediction "${prior_gqba}" "${GQBA_ROOT}/validation_predictions.tar.gz"
  for control in "${joint_control}" "${prior_gqba}"; do
    if [[ ! -f "${control}/summary.json" \
        || ! -f "${control}/validation_predictions.csv" ]]; then
      echo "Missing locked control: ${control}" >&2
      echo "Restore the Phase-A and GQBA prediction CSV files before running GSRA." >&2
      exit 1
    fi
  done
  RUN_SPECS+=("joint_multiview_ce=${joint_control}")
  RUN_SPECS+=("gqba_odd_even=${prior_gqba}")
  run_candidate gated_spectral_control spectral_capacity_control 0.0 0.0 0.0 "${seed}"
  run_candidate gqba_gated_ce gqba_odd_even 0.0 0.0 0.0 "${seed}"
  run_candidate gsra gqba_odd_even 1.0 0.1 0.1 "${seed}"
done

"${GAUGEEG[@]}" gsra-audit \
  --runs "${RUN_SPECS[@]}" \
  --expected-seeds "${SEED_ARRAY[@]}" \
  --output-dir "${OUTPUT_ROOT}/aggregate"

tar --no-recursion -czf \
  "${OUTPUT_ROOT}/validation_predictions.tar.gz" \
  "${PREDICTION_FILES[@]}"

echo "Completed the locked GSRA development screen."
python -m json.tool "${OUTPUT_ROOT}/aggregate/gsra_summary.json"

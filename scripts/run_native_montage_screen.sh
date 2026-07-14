#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_native_montage_screen.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

E7A_SUMMARY="outputs/reve_montage_screen_s7/montage_screen_summary.json"
if [[ ! -f "${E7A_SUMMARY}" ]]; then
  echo "Missing ${E7A_SUMMARY}. Retain the E7a result that motivated this correction." >&2
  exit 1
fi

test_views=(
  car
  "native32@car" "native16@car" "native8@car"
  "native32@cz" "native16@cz" "native8@cz"
  "native_drop_left_motor@cz" "native_drop_right_motor@cz"
)

run_if_missing() {
  local output_dir="$1"
  local defense="$2"
  if [[ -f "${output_dir}/metrics.csv" && -f "${output_dir}/predictions.csv" ]]; then
    echo "Reusing completed run: ${output_dir}"
  else
    "${GAUGEEG[@]}" run \
      --config "${CONFIG}" \
      --device "${DEVICE}" \
      --probe-seed 7 \
      --reference-seed 7 \
      --training-views car \
      --test-views "${test_views[@]}" \
      --defenses "${defense}" \
      --probe-objective car_only \
      --consistency-weight 0 \
      --output-dir "${output_dir}"
  fi
}

run_if_missing outputs/reve_native_montage_car_only_s7 none
run_if_missing outputs/reve_native_montage_car_canonicalize_s7 car_canonicalize

"${GAUGEEG[@]}" native-montage-screen \
  --baseline outputs/reve_native_montage_car_only_s7 \
  --canonical outputs/reve_native_montage_car_canonicalize_s7 \
  --primary-view "native16@cz" \
  --output-dir outputs/reve_native_montage_screen_s7

echo "Completed E7b. Benchmark validity decision:"
echo "  outputs/reve_native_montage_screen_s7/native_montage_screen_summary.json"
python -m json.tool outputs/reve_native_montage_screen_s7/native_montage_screen_summary.json

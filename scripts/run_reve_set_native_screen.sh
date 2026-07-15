#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_set_native_screen.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

if [[ ! -f outputs/reve_native_montage_screen_s7/native_montage_screen_summary.json ]]; then
  echo "Missing the E7b summary. Pull or restore the result that motivated E7c first." >&2
  exit 1
fi

run_clean_if_missing() {
  local queries="$1"
  local output_dir="outputs/reve_set_clean_q${queries}_s7"
  if [[ -f "${output_dir}/metrics.csv" && -f "${output_dir}/predictions.csv" ]]; then
    echo "Reusing completed clean run: ${output_dir}"
  else
    "${GAUGEEG[@]}" run \
      --config "${CONFIG}" \
      --device "${DEVICE}" \
      --set-queries "${queries}" \
      --test-views car \
      --defenses none \
      --output-dir "${output_dir}"
  fi
}

for queries in 4 8 16; do
  run_clean_if_missing "${queries}"
done

"${GAUGEEG[@]}" select-set-head \
  --runs \
    outputs/reve_set_clean_q4_s7 \
    outputs/reve_set_clean_q8_s7 \
    outputs/reve_set_clean_q16_s7 \
  --expected-queries 4 8 16 \
  --clean-gate 0.45 \
  --output-dir outputs/reve_set_head_selection_s7

SELECTION=outputs/reve_set_head_selection_s7/set_head_selection.json
python -m json.tool "${SELECTION}"
read -r SELECTED_QUERIES CLEAN_PASSED < <(
  python -c \
    'import json,sys; x=json.load(open(sys.argv[1])); print(x["selected_queries"], int(x["clean_gate_passed"]))' \
    "${SELECTION}"
)
if [[ "${CLEAN_PASSED}" != "1" ]]; then
  echo "E7c stopped: validation-selected set head did not pass clean test BAcc >= 0.45." >&2
  echo "Push the three reve_set_clean_q*_s7 directories and reve_set_head_selection_s7." >&2
  exit 0
fi

test_views=(
  car
  "native32@car" "native16@car" "native8@car"
  "native32@cz" "native16@cz" "native8@cz"
  "native_drop_left_motor@cz" "native_drop_right_motor@cz"
)

run_native_if_missing() {
  local output_dir="$1"
  local defense="$2"
  if [[ -f "${output_dir}/metrics.csv" && -f "${output_dir}/predictions.csv" ]]; then
    echo "Reusing completed native run: ${output_dir}"
  else
    "${GAUGEEG[@]}" run \
      --config "${CONFIG}" \
      --device "${DEVICE}" \
      --set-queries "${SELECTED_QUERIES}" \
      --test-views "${test_views[@]}" \
      --defenses "${defense}" \
      --output-dir "${output_dir}"
  fi
}

BASELINE="outputs/reve_set_native_q${SELECTED_QUERIES}_none_s7"
CANONICAL="outputs/reve_set_native_q${SELECTED_QUERIES}_car_canonicalize_s7"
run_native_if_missing "${BASELINE}" none
run_native_if_missing "${CANONICAL}" car_canonicalize

"${GAUGEEG[@]}" native-montage-screen \
  --baseline "${BASELINE}" \
  --canonical "${CANONICAL}" \
  --primary-view "native16@cz" \
  --output-dir outputs/reve_set_native_montage_screen_s7

echo "Completed E7c. Push these result directories:"
echo "  outputs/reve_set_clean_q4_s7"
echo "  outputs/reve_set_clean_q8_s7"
echo "  outputs/reve_set_clean_q16_s7"
echo "  outputs/reve_set_head_selection_s7"
echo "  ${BASELINE}"
echo "  ${CANONICAL}"
echo "  outputs/reve_set_native_montage_screen_s7"
python -m json.tool outputs/reve_set_native_montage_screen_s7/native_montage_screen_summary.json

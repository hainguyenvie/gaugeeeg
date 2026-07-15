#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_set_native_screen.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

SELECTION=outputs/reve_set_head_selection_s7/set_head_selection.json
NATIVE_RUN=outputs/reve_set_native_q4_none_s7
if [[ ! -f "${SELECTION}" || ! -f "${NATIVE_RUN}/predictions.csv" ]]; then
  echo "Missing E7c selection/native results. Pull commit 8ce60a1 before running E7d." >&2
  exit 1
fi

SELECTED_QUERIES=$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_queries"])' "${SELECTION}"
)
if [[ "${SELECTED_QUERIES}" != "4" ]]; then
  echo "E7d expects the already-frozen E7c q4 head, observed q${SELECTED_QUERIES}." >&2
  exit 1
fi

FULL_RUN="outputs/reve_set_full_reference_q${SELECTED_QUERIES}_none_s7"
if [[ -f "${FULL_RUN}/metrics.csv" && -f "${FULL_RUN}/predictions.csv" ]]; then
  echo "Reusing completed full-reference run: ${FULL_RUN}"
else
  "${GAUGEEG[@]}" run \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --set-queries "${SELECTED_QUERIES}" \
    --training-views car \
    --test-views car cz pz fcz \
    --defenses none \
    --probe-objective car_only \
    --consistency-weight 0 \
    --output-dir "${FULL_RUN}"
fi

CLOSURE=outputs/reve_set_reference_closure_s7
"${GAUGEEG[@]}" reference-closure \
  --full-run "${FULL_RUN}" \
  --native-run "${NATIVE_RUN}" \
  --selection "${SELECTION}" \
  --output-dir "${CLOSURE}"

echo "Completed E7d. Push these result directories:"
echo "  ${FULL_RUN}"
echo "  ${CLOSURE}"
python -m json.tool "${CLOSURE}/reference_closure_summary.json"

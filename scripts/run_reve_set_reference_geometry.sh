#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_set_native_screen.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

SELECTION=outputs/reve_set_head_selection_s7/set_head_selection.json
E7D_FULL=outputs/reve_set_full_reference_q4_none_s7
if [[ ! -f "${SELECTION}" || ! -f "${E7D_FULL}/predictions.csv" ]]; then
  echo "Missing E7c selection or E7d full-reference results." >&2
  exit 1
fi

SELECTED_QUERIES=$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_queries"])' "${SELECTION}"
)
if [[ "${SELECTED_QUERIES}" != "4" ]]; then
  echo "E7e expects the frozen E7c q4 head, observed q${SELECTED_QUERIES}." >&2
  exit 1
fi

test_views=(
  car
  "native32@car" "native32@cz" "native32@pz" "native32@fz"
  "native16@car" "native16@cz" "native16@pz" "native16@fz"
)

GEOMETRY_RUN="outputs/reve_set_native_reference_geometry_q4_s7"
if [[ -f "${GEOMETRY_RUN}/metrics.csv" && -f "${GEOMETRY_RUN}/predictions.csv" ]]; then
  echo "Reusing completed native-reference geometry run: ${GEOMETRY_RUN}"
else
  "${GAUGEEG[@]}" run \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --set-queries "${SELECTED_QUERIES}" \
    --training-views car \
    --test-views "${test_views[@]}" \
    --defenses none \
    --probe-objective car_only \
    --consistency-weight 0 \
    --output-dir "${GEOMETRY_RUN}"
fi

AUDIT=outputs/reve_set_reference_geometry_audit_s7
"${GAUGEEG[@]}" reference-geometry \
  --run "${GEOMETRY_RUN}" \
  --e7d-full-run "${E7D_FULL}" \
  --selection "${SELECTION}" \
  --output-dir "${AUDIT}"

echo "Completed E7e. Push these result directories:"
echo "  ${GEOMETRY_RUN}"
echo "  ${AUDIT}"
python -m json.tool "${AUDIT}/reference_geometry_summary.json"

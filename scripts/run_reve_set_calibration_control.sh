#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_set_calibration_control_q4.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

SELECTION=outputs/reve_set_head_selection_s7/set_head_selection.json
E7E_RUN=outputs/reve_set_native_reference_geometry_q4_s7
if [[ ! -f "${SELECTION}" || ! -f "${E7E_RUN}/predictions.csv" ]]; then
  echo "Missing E7c selection or E7e predictions. Run E7e first." >&2
  exit 1
fi

SELECTED_QUERIES=$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_queries"])' "${SELECTION}"
)
if [[ "${SELECTED_QUERIES}" != "4" ]]; then
  echo "E8 expects the frozen E7c q4 head, observed q${SELECTED_QUERIES}." >&2
  exit 1
fi

LOGIT_RUN=outputs/reve_set_calibration_logits_q4_s7
if [[ -f "${LOGIT_RUN}/predictions.csv" && -f "${LOGIT_RUN}/validation_predictions.csv" ]]; then
  echo "Reusing completed E8 logit run: ${LOGIT_RUN}"
else
  "${GAUGEEG[@]}" run \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --set-queries "${SELECTED_QUERIES}" \
    --output-dir "${LOGIT_RUN}"
fi

AUDIT=outputs/reve_set_calibration_control_s7
"${GAUGEEG[@]}" calibration-control \
  --validation-predictions "${LOGIT_RUN}/validation_predictions.csv" \
  --test-predictions "${LOGIT_RUN}/predictions.csv" \
  --baseline-predictions "${E7E_RUN}/predictions.csv" \
  --output-dir "${AUDIT}"

echo "Completed E8. Push these result directories:"
echo "  ${LOGIT_RUN}"
echo "  ${AUDIT}"
python -m json.tool "${AUDIT}/calibration_summary.json"

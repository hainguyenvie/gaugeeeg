#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_set_bias_manifold_q4.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

SELECTION=outputs/reve_set_head_selection_s7/set_head_selection.json
E8_RUN=outputs/reve_set_calibration_logits_q4_s7
if [[ ! -f "${SELECTION}" || ! -f "${E8_RUN}/validation_predictions.csv" ]]; then
  echo "Missing E7c q4 selection or E8 validation logits. Run E8 first." >&2
  exit 1
fi

SELECTED_QUERIES=$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_queries"])' "${SELECTION}"
)
if [[ "${SELECTED_QUERIES}" != "4" ]]; then
  echo "E9 expects the frozen E7c q4 head, observed q${SELECTED_QUERIES}." >&2
  exit 1
fi

LOGIT_RUN=outputs/reve_set_bias_manifold_logits_q4_s7
if [[ -f "${LOGIT_RUN}/validation_predictions.csv" && -f "${LOGIT_RUN}/summary.json" ]]; then
  echo "Reusing completed E9 validation grid: ${LOGIT_RUN}"
else
  "${GAUGEEG[@]}" run \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --set-queries "${SELECTED_QUERIES}" \
    --output-dir "${LOGIT_RUN}"
fi

AUDIT=outputs/reve_set_bias_manifold_audit_s7
"${GAUGEEG[@]}" bias-manifold \
  --validation-predictions "${LOGIT_RUN}/validation_predictions.csv" \
  --e8-validation-predictions "${E8_RUN}/validation_predictions.csv" \
  --output-dir "${AUDIT}"

echo "Completed E9. Push these result directories (exclude .pt checkpoints):"
echo "  ${LOGIT_RUN}"
echo "  ${AUDIT}"
python -m json.tool "${AUDIT}/bias_manifold_summary.json"

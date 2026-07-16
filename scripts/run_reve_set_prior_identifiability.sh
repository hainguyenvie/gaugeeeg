#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

E9_LOGITS=outputs/reve_set_bias_manifold_logits_q4_s7/validation_predictions.csv
E9_SUMMARY=outputs/reve_set_bias_manifold_audit_s7/bias_manifold_summary.json
if [[ ! -f "${E9_LOGITS}" || ! -f "${E9_SUMMARY}" ]]; then
  echo "Missing E9 validation logits or audit summary. Run E9 first." >&2
  exit 1
fi

python - "${E9_SUMMARY}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
required = [
    "fit_evaluation_subjects_disjoint",
    "unlabeled_logit_conditioning_supported",
]
failed = [key for key in required if not summary.get(key, False)]
if failed:
    raise SystemExit(f"E9 prerequisite checks failed: {failed}")
PY

OUTPUT=outputs/reve_set_prior_identifiability_audit_s7
"${GAUGEEG[@]}" prior-identifiability \
  --validation-predictions "${E9_LOGITS}" \
  --output-dir "${OUTPUT}"

echo "Completed CPU-only E11: ${OUTPUT}"
python -m json.tool "${OUTPUT}/prior_identifiability_summary.json"

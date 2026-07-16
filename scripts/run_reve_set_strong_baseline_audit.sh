#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

E12_OUTPUT=outputs/reve_set_class_safeguard_audit_s7
E12_SUMMARY=${E12_OUTPUT}/class_safeguard_summary.json
E12_METRICS=${E12_OUTPUT}/class_safeguard_metrics.csv
if [[ ! -f "${E12_SUMMARY}" || ! -f "${E12_METRICS}" ]]; then
  echo "Missing the complete E12 output. Run E12 first." >&2
  exit 1
fi

python - "${E12_SUMMARY}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
required = [
    "source_adaptation_subjects_disjoint",
    "source_evaluation_subjects_disjoint",
    "adaptation_evaluation_subjects_disjoint",
    "safeguard_supported_for_repeated_seed_confirmation",
]
failed = [key for key in required if not summary.get(key, False)]
if failed:
    raise SystemExit(f"E12 prerequisite checks failed: {failed}")
PY

OUTPUT=outputs/reve_set_strong_baseline_audit_s7
"${GAUGEEG[@]}" strong-baseline-audit \
  --e12-output "${E12_OUTPUT}" \
  --output-dir "${OUTPUT}"

echo "Completed CPU-only E13: ${OUTPUT}"
python -m json.tool "${OUTPUT}/strong_baseline_summary.json"

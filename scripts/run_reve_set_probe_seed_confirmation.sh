#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_set_probe_confirmation_q4.yaml}"
DEVICE="${DEVICE:-cuda}"
SEED_LIST="${CONFIRMATION_SEEDS:-21 42}"
read -r -a SEEDS <<< "${SEED_LIST}"

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

FROZEN_E13=outputs/reve_set_strong_baseline_audit_s7/strong_baseline_summary.json
SELECTION=outputs/reve_set_head_selection_s7/set_head_selection.json
if [[ ! -f "${FROZEN_E13}" || ! -f "${SELECTION}" ]]; then
  echo "Missing the frozen E13 gate or E7c head selection." >&2
  exit 1
fi

python - "${FROZEN_E13}" "${SELECTION}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    gate = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    selection = json.load(handle)
failed = []
if not gate.get("mean_method_ready_for_new_seed_confirmation", False):
    failed.append("E13 mean-method readiness")
if gate.get("class_uniform_method_ready_for_new_seed_confirmation", True):
    failed.append("E13 class-uniform rejection")
if selection.get("selected_queries") != 4:
    failed.append("frozen q4 head")
if failed:
    raise SystemExit(f"Frozen E14 prerequisites failed: {failed}")
PY

if [[ "${#SEEDS[@]}" -lt 2 ]]; then
  echo "E14 requires at least two untouched probe seeds." >&2
  exit 1
fi

LOGIT_RUNS=()
E12_RUNS=()
for seed in "${SEEDS[@]}"; do
  if [[ "${seed}" == "7" ]]; then
    echo "Seed 7 is exploratory and cannot enter E14 confirmation." >&2
    exit 1
  fi

  LOGIT_RUN="outputs/reve_set_probe_confirmation_logits_q4_s${seed}"
  E12_RUN="outputs/reve_set_class_safeguard_audit_s${seed}"
  LOGIT_RUNS+=("${LOGIT_RUN}")
  E12_RUNS+=("${E12_RUN}")

  if [[ -f "${LOGIT_RUN}/summary.json" \
      && -f "${LOGIT_RUN}/validation_predictions.csv" ]]; then
    echo "Reusing completed untouched-probe logit run: ${LOGIT_RUN}"
  else
    echo "Training q4 probe seed ${seed} with test subjects untouched"
    "${GAUGEEG[@]}" run \
      --config "${CONFIG}" \
      --device "${DEVICE}" \
      --probe-seed "${seed}" \
      --reference-seed 7 \
      --set-queries 4 \
      --output-dir "${LOGIT_RUN}"
  fi

  python - "${LOGIT_RUN}/summary.json" "${seed}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
seed = int(sys.argv[2])
checks = {
    "probe_seed": summary.get("probe_seed") == seed,
    "reference_seed": summary.get("reference_seed") == 7,
    "validation_predictions_only": summary.get("validation_predictions_only") is True,
    "prediction_split": summary.get("prediction_split") == "audit",
    "split_disjointness": summary.get("all_subject_splits_pairwise_disjoint") is True,
    "test_untouched": (
        summary.get("physionet_test_subjects_used_for_fitting_or_scoring") is False
    ),
}
failed = [key for key, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Probe seed {seed} protocol checks failed: {failed}")
PY

  rerun_e12=true
  if [[ -f "${E12_RUN}/class_safeguard_summary.json" \
      && -f "${E12_RUN}/class_safeguard_metrics.csv" ]]; then
    if python - "${E12_RUN}/class_safeguard_summary.json" "${seed}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
raise SystemExit(0 if summary.get("probe_seed") == int(sys.argv[2]) else 1)
PY
    then
      rerun_e12=false
      echo "Reusing completed E12 for probe seed ${seed}: ${E12_RUN}"
    fi
  fi
  if [[ "${rerun_e12}" == "true" ]]; then
    echo "Running strict E12 audit for probe seed ${seed}"
    "${GAUGEEG[@]}" class-safeguard \
      --validation-predictions "${LOGIT_RUN}/validation_predictions.csv" \
      --output-dir "${E12_RUN}"
  fi
done

OUTPUT=outputs/reve_set_probe_seed_confirmation
"${GAUGEEG[@]}" confirm-probe-seeds \
  --frozen-e13-summary "${FROZEN_E13}" \
  --logit-runs "${LOGIT_RUNS[@]}" \
  --e12-runs "${E12_RUNS[@]}" \
  --output-dir "${OUTPUT}"

echo "Completed E14. Push these directories (feature cache is excluded):"
for path in "${LOGIT_RUNS[@]}" "${E12_RUNS[@]}" "${OUTPUT}"; do
  echo "  ${path}"
done
python -m json.tool "${OUTPUT}/probe_seed_confirmation_summary.json"

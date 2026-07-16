#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_set_operator_consistency_q4.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

E14=outputs/reve_set_probe_seed_confirmation/probe_seed_confirmation_summary.json
if [[ ! -f "${E14}" ]]; then
  echo "Missing the terminal E14 summary: ${E14}" >&2
  exit 1
fi

python - "${E14}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
checks = {
    "stage": summary.get("stage") == "E14 untouched-probe-seed mean-method confirmation",
    "negative_gate": summary.get("mean_method_new_seed_confirmation_supported") is False,
    "test_untouched": summary.get("physionet_test_subjects_used") is False,
    "recommendation": summary.get("next_method_recommendation")
    == "do_not_tune_on_audit_subjects_revisit_source_only_method",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Frozen E15 prerequisites failed: {failed}")
PY

CAR_ONLY=outputs/reve_set_operator_screen_car_only_s7
MULTI_VIEW=outputs/reve_set_operator_screen_multi_view_ce_s7
OPERATOR=outputs/reve_set_operator_screen_consistency_s7
OUTPUT=outputs/reve_set_operator_consistency_screen_s7

run_arm() {
  local name="$1"
  local output_dir="$2"
  local objective="$3"
  local weight="$4"
  shift 4
  local training_views=("$@")

  if [[ -f "${output_dir}/summary.json" \
      && -f "${output_dir}/validation_predictions.csv" ]]; then
    if python - \
      "${output_dir}/summary.json" \
      "${objective}" \
      "${weight}" \
      "${training_views[@]}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
objective = sys.argv[2]
weight = float(sys.argv[3])
training_views = sys.argv[4:]
audit_views = [
    "car",
    "native32@car",
    "native16@car",
    "native32@Cz",
    "native16@Cz",
    "native32@Pz",
    "native16@Pz",
]
checks = {
    "objective": summary.get("probe_objective") == objective,
    "training_views": summary.get("training_views") == training_views,
    "audit_views": summary.get("validation_prediction_views") == audit_views,
    "consistency_weight": summary.get("consistency_weight") == weight,
    "probe_seed": summary.get("probe_seed") == 7,
    "reference_seed": summary.get("reference_seed") == 7,
    "q4": summary.get("set_queries") == 4,
    "audit_only": summary.get("validation_predictions_only") is True,
    "split": summary.get("prediction_split") == "audit",
    "disjoint": summary.get("all_subject_splits_pairwise_disjoint") is True,
    "test_untouched": summary.get("physionet_test_subjects_used_for_fitting_or_scoring")
    is False,
}
if objective == "operator_consistency":
    checks["view_weights"] = summary.get("consistency_view_weights") == [
        0.0,
        0.5,
        1.0,
    ]
raise SystemExit(0 if all(checks.values()) else 1)
PY
    then
      echo "Reusing completed ${name}: ${output_dir}"
      return
    fi
  fi

  echo "Training E15 ${name} (probe seed 7; reserved test untouched)"
  "${GAUGEEG[@]}" run \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --probe-seed 7 \
    --reference-seed 7 \
    --set-queries 4 \
    --probe-objective "${objective}" \
    --consistency-weight "${weight}" \
    --training-views "${training_views[@]}" \
    --output-dir "${output_dir}"
}

run_arm "CAR-only control" "${CAR_ONLY}" car_only 0.0 car
run_arm \
  "multi-montage CE control" \
  "${MULTI_VIEW}" \
  multi_view_ce \
  0.0 \
  car "native32@car" "native16@car"
run_arm \
  "CAR-teacher operator consistency" \
  "${OPERATOR}" \
  operator_consistency \
  1.0 \
  car "native32@car" "native16@car"

"${GAUGEEG[@]}" operator-consistency-audit \
  --e14-summary "${E14}" \
  --car-only-run "${CAR_ONLY}" \
  --multi-view-run "${MULTI_VIEW}" \
  --operator-run "${OPERATOR}" \
  --output-dir "${OUTPUT}"

echo "Completed E15 development screen. Reserved test subjects 90--109 remain untouched."
echo "Push these four directories (feature cache and checkpoints remain excluded):"
printf '  %s\n' "${CAR_ONLY}" "${MULTI_VIEW}" "${OPERATOR}" "${OUTPUT}"
python -m json.tool "${OUTPUT}/operator_consistency_summary.json"

#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_montage_screen.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

LAMBDA_SUMMARY="outputs/reve_consistency_lambda_ablation/lambda_ablation_summary.json"
if [[ ! -f "${LAMBDA_SUMMARY}" ]]; then
  echo "Missing ${LAMBDA_SUMMARY}. Complete the validation-only lambda stage first." >&2
  exit 1
fi
python - "${LAMBDA_SUMMARY}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if float(summary["selected_lambda"]) != 10.0:
    raise SystemExit(
        f"This predeclared screen expects validation-selected lambda=10; found {summary['selected_lambda']}"
    )
if bool(summary.get("selection_uses_target_view", True)):
    raise SystemExit("Refusing to continue because lambda selection used the held-out target view")
PY

test_views=(
  car
  "sparse32@car" "sparse16@car" "sparse8@car"
  "sparse32@cz" "sparse16@cz" "sparse8@cz"
  "drop_left_motor@cz" "drop_right_motor@cz"
)

run_if_missing() {
  local output_dir="$1"
  shift
  if [[ -f "${output_dir}/metrics.csv" && -f "${output_dir}/predictions.csv" ]]; then
    echo "Reusing completed run: ${output_dir}"
  else
    "${GAUGEEG[@]}" run \
      --config "${CONFIG}" \
      --device "${DEVICE}" \
      --probe-seed 7 \
      --reference-seed 7 \
      --test-views "${test_views[@]}" \
      --output-dir "${output_dir}" \
      "$@"
  fi
}

# Baseline 1: CAR-only probe, no protection.
run_if_missing outputs/reve_montage_car_only_s7 \
  --training-views car \
  --defenses none \
  --probe-objective car_only \
  --consistency-weight 0

# Baseline 2: exact positive control for full-channel reference changes. It is
# intentionally no longer exact once channels are unobserved.
run_if_missing outputs/reve_montage_car_canonicalize_s7 \
  --training-views car \
  --defenses car_canonicalize \
  --probe-objective car_only \
  --consistency-weight 0

# Baseline 3: full-montage reference augmentation.
run_if_missing outputs/reve_montage_multiview_ce_s7 \
  --training-views car pz fcz \
  --defenses none \
  --probe-objective multi_view_ce \
  --consistency-weight 0

# Existing selected method: lambda is fixed before any montage target is read.
run_if_missing outputs/reve_montage_rule_consistency_lam10_s7 \
  --training-views car pz fcz \
  --defenses none \
  --probe-objective rule_consistency \
  --consistency-weight 10

"${GAUGEEG[@]}" montage-screen \
  --car-only outputs/reve_montage_car_only_s7 \
  --canonical outputs/reve_montage_car_canonicalize_s7 \
  --augmentation outputs/reve_montage_multiview_ce_s7 \
  --consistency outputs/reve_montage_rule_consistency_lam10_s7 \
  --primary-view "sparse16@cz" \
  --target-class 0 \
  --selected-lambda 10 \
  --output-dir outputs/reve_montage_screen_s7

echo "Completed E7a. Primary decision file:"
echo "  outputs/reve_montage_screen_s7/montage_screen_summary.json"
python -m json.tool outputs/reve_montage_screen_s7/montage_screen_summary.json

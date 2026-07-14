#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/reve_consistency_screen.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required artifact: $1" >&2
    echo "Run and retain the previous multi-seed stage before this ablation." >&2
    exit 1
  fi
}

seeds=(7 21 42)
new_weights=(0.3 3.0 10.0)
baselines=()
runs=()

for seed in "${seeds[@]}"; do
  baseline="outputs/reve_statistical_audit_s${seed}"
  augmentation="outputs/reve_multiview_ce_s${seed}"
  lambda_one="outputs/reve_rule_consistency_s${seed}"
  require_file "${baseline}/predictions.csv"
  require_file "${augmentation}/predictions.csv"
  require_file "${lambda_one}/predictions.csv"
  baselines+=("${baseline}")
  runs+=("${augmentation}" "${lambda_one}")

  for weight in "${new_weights[@]}"; do
    tag="${weight//./p}"
    output_dir="outputs/reve_rule_consistency_lam${tag}_s${seed}"
    if [[ -f "${output_dir}/metrics.csv" && -f "${output_dir}/predictions.csv" ]]; then
      echo "Reusing completed run: seed=${seed}, lambda=${weight}"
    else
      echo "Running rule consistency: seed=${seed}, lambda=${weight}"
      "${GAUGEEG[@]}" run \
        --config "${CONFIG}" \
        --device "${DEVICE}" \
        --probe-seed "${seed}" \
        --reference-seed 7 \
        --probe-objective rule_consistency \
        --consistency-weight "${weight}" \
        --output-dir "${output_dir}"
    fi
    runs+=("${output_dir}")
  done
done

"${GAUGEEG[@]}" lambda-ablation \
  --baselines "${baselines[@]}" \
  --runs "${runs[@]}" \
  --expected-lambdas 0 0.3 1 3 10 \
  --target-view cz \
  --target-class 0 \
  --output-dir outputs/reve_consistency_lambda_ablation

echo "Completed. Validation-only selection and final evidence:"
echo "  outputs/reve_consistency_lambda_ablation/lambda_ablation_summary.json"
python -m json.tool outputs/reve_consistency_lambda_ablation/lambda_ablation_summary.json

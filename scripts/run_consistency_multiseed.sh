#!/usr/bin/env bash
set -euo pipefail

# Frozen REVE features are shared across runs. This script intentionally does
# not use --force-recompute, so seeds 21/42 train new probes without re-encoding EEG.
CONFIG="${CONFIG:-configs/reve_consistency_screen.yaml}"
DEVICE="${DEVICE:-cuda}"

# Always execute the just-pulled source tree. This avoids accidentally using an
# older non-editable GaugeEEG installation that happens to be on PATH.
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
GAUGEEG=(python -m gaugeeeg.cli)

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required artifact: $1" >&2
    echo "Pull the committed seed-7 results or generate the corresponding run first." >&2
    exit 1
  fi
}

for seed in 7 21 42; do
  require_file "outputs/reve_statistical_audit_s${seed}/metrics.csv"
  require_file "outputs/reve_statistical_audit_s${seed}/predictions.csv"
done
require_file "outputs/reve_multiview_ce_s7/predictions.csv"
require_file "outputs/reve_rule_consistency_s7/predictions.csv"

# Regenerate seed 7 comparison with the new direct paired-method bootstrap.
"${GAUGEEG[@]}" compare-methods \
  --baseline outputs/reve_statistical_audit_s7 \
  --augmentation outputs/reve_multiview_ce_s7 \
  --consistency outputs/reve_rule_consistency_s7 \
  --output-dir outputs/reve_consistency_comparison_s7

for seed in 21 42; do
  echo "Running multi-view CE probe for seed ${seed}"
  "${GAUGEEG[@]}" run \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --probe-seed "$seed" \
    --reference-seed 7 \
    --probe-objective multi_view_ce \
    --output-dir "outputs/reve_multiview_ce_s${seed}"

  echo "Running rule-consistency probe for seed ${seed}"
  "${GAUGEEG[@]}" run \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --probe-seed "$seed" \
    --reference-seed 7 \
    --probe-objective rule_consistency \
    --consistency-weight 1.0 \
    --output-dir "outputs/reve_rule_consistency_s${seed}"

  "${GAUGEEG[@]}" compare-methods \
    --baseline "outputs/reve_statistical_audit_s${seed}" \
    --augmentation "outputs/reve_multiview_ce_s${seed}" \
    --consistency "outputs/reve_rule_consistency_s${seed}" \
    --output-dir "outputs/reve_consistency_comparison_s${seed}"
done

"${GAUGEEG[@]}" aggregate-methods \
  --baselines \
    outputs/reve_statistical_audit_s7 \
    outputs/reve_statistical_audit_s21 \
    outputs/reve_statistical_audit_s42 \
  --augmentations \
    outputs/reve_multiview_ce_s7 \
    outputs/reve_multiview_ce_s21 \
    outputs/reve_multiview_ce_s42 \
  --consistencies \
    outputs/reve_rule_consistency_s7 \
    outputs/reve_rule_consistency_s21 \
    outputs/reve_rule_consistency_s42 \
  --output-dir outputs/reve_consistency_comparison_multiseed

echo "Completed. Primary decision file:"
echo "  outputs/reve_consistency_comparison_multiseed/aggregate_method_summary.json"
python -m json.tool \
  outputs/reve_consistency_comparison_multiseed/aggregate_method_summary.json

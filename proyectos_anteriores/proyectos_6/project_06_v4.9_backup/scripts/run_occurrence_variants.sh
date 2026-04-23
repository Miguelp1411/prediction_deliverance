#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${OCC_MONITOR_NAME:=count_mae}"
: "${OCC_MONITOR_MIN_DELTA:=0.01}"

for baseline in 1.5 1.0; do
  echo "==== Ejecutando con OCC_SEASONAL_BASELINE_LOGIT=${baseline}, monitor=${OCC_MONITOR_NAME} ===="
  OCC_SEASONAL_BASELINE_LOGIT="$baseline" \
  OCC_MONITOR_NAME="$OCC_MONITOR_NAME" \
  OCC_MONITOR_MIN_DELTA="$OCC_MONITOR_MIN_DELTA" \
  python3 train.py
  out_dir="reports/baseline_${baseline//./_}_${OCC_MONITOR_NAME}"
  mkdir -p "$out_dir"
  cp -f reports/training_report.json "$out_dir/training_report.json"
  cp -f checkpoints/occurrence_model.pt "$out_dir/occurrence_model.pt"
  cp -f checkpoints/temporal_model.pt "$out_dir/temporal_model.pt"
done

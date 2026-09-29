#!/usr/bin/env bash
# One training stage of the chess fine-tune: one epoch, full weights, from INITIAL_CHECKPOINT with a fresh optimizer.
# Checkpoints are selected on DATA_DIR/dev.jsonl and calibrated on DATA_DIR/calibration.jsonl. Run from the repository root.
# Writes runs/chess/RUN (event log, evaluations) and checkpoints/chess/RUN (step-* folders and "selected").
set -euo pipefail
if [[ "$#" -ne 3 ]]; then
  echo "Usage: examples/chess/train.sh RUN DATA_DIR INITIAL_CHECKPOINT" >&2
  exit 2
fi
run=$1 data=$2 initial=$3
for file in train.jsonl dev.jsonl calibration.jsonl; do
  [[ -f "$data/$file" ]] || { echo "Missing $data/$file (build it with examples/chess/build_rows.py)" >&2; exit 1; }
done
[[ -f "$initial/decision_config.json" ]] || { echo "$initial is not a Jeff checkpoint (no decision_config.json)" >&2; exit 1; }
mkdir -p "runs/chess/$run"
export JEFF_EVENTS="runs/chess/$run/events.jsonl" PYTHONUNBUFFERED=1
exec uv run jeff-train --train "$data/train.jsonl" --development "$data/dev.jsonl" --temperature "$data/calibration.jsonl" \
  --run "runs/chess/$run" --output "checkpoints/chess/$run" \
  --base-model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --initial-checkpoint "$initial" \
  --epochs 1 --seed 20260920 --lr 5e-6 --weight-decay 0.01 --batch-size 32 --effective-batch-size 256 \
  --token-budget 8192 --max-length 8192 --cpu-threads 16 --eval-every 40 --public-eval-every 1000000 --resume-every 50

#!/usr/bin/env bash
# Full-weight SFT of a Qwen3.5 student with the jeff trainer. Checkpoints are selected on DATA_DIR/dev.jsonl only; the panel is never used.
set -euo pipefail
if [[ "$#" -lt 7 ]]; then
  echo "Usage: scripts/train.sh RUN TRAIN_JSONL DATA_DIR LR EVAL_EVERY MODEL REVISION [extra jeff-train args]" >&2
  exit 2
fi
run=$1 train=$2 data=$3 lr=$4 every=$5 model=$6 revision=$7
shift 7
export JEFF_EVENTS="runs/$run/events.jsonl" PYTHONUNBUFFERED=1
exec uv run jeff-train --train "$train" --development "$data/dev.jsonl" --temperature "$data/calibration.jsonl" \
  --run "runs/$run" --output "checkpoints/$run" \
  --base-model "$model" --revision "$revision" \
  --epochs 1 --seed 20260920 --lr "$lr" --weight-decay 0.01 --batch-size 32 --effective-batch-size 256 \
  --token-budget 8192 --max-length 8192 --cpu-threads 16 \
  --eval-every "$every" --public-eval-every 1000000 --resume-every 50 "$@"

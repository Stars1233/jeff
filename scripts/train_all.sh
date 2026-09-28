#!/usr/bin/env bash
# The recipe behind the released models: build the training set from every source, check lengths, then train and score
# the 0.8B, 2B and Gemma 4 E2B students (ONE epoch each), each scored on the panel, JevBench, documents and voice sets.
#
# Usage: scripts/train_all.sh SYNTHETIC_JSONL
#   SYNTHETIC_JSONL: synthetic questions from `jeff-generate run` + `jeff-generate finalize` (needs JEFF_TEACHER_URL).
# Inputs built beforehand (see README): data/extra (jeff-extra), data/benchmark-train (jeff-extra --only ragtruth_train
# winogrande_train), data/documents, data/voice, data/probability (jeff-probability), data/panel.jsonl (jeff-panel),
# data/jevbench-hard.jsonl (jeff-jevbench). Learning rates: 5e-6 (0.8B), 1e-5 (2B), 2e-5 (Gemma), from short sweeps.
set -euo pipefail
cd "$(dirname "$0")/.."
synthetic=${1:?usage: scripts/train_all.sh SYNTHETIC_JSONL}
data=data/mix
uv run jeff-mix --public-only --panel-layout --adversarial --escape --extra data/extra/train.jsonl \
  data/benchmark-train/train.jsonl data/documents/train.jsonl data/voice/train.jsonl data/probability/train.jsonl \
  "$synthetic" --size 1000000 --out "$data"
score() {  # RUN
  local ckpt=checkpoints/$1/selected
  uv run jeff-evaluate --data data/panel.jsonl --output "runs/eval/$1-calibrated.json" --local --checkpoint "$ckpt" --batch-size 16
  uv run jeff-evaluate --data data/jevbench-hard.jsonl --output "runs/eval/jevbench/$1-calibrated.json" --local --checkpoint "$ckpt" --batch-size 4
  uv run jeff-evaluate --data data/documents/check.jsonl --output "runs/eval/documents/$1-calibrated.json" --local --checkpoint "$ckpt" --batch-size 8
  uv run jeff-evaluate --data data/voice/test.jsonl --output "runs/eval/voice/$1-calibrated.json" --local --checkpoint "$ckpt" --batch-size 8
}
while read -r tag model revision lr; do
  run=$tag-$(date +%Y%m%d-%H%M)
  scripts/train.sh "$run" "$data/public.jsonl" "$data" "$lr" 40 "$model" "$revision" --epochs 1
  score "$run"
done <<MODELS
0.8b Qwen/Qwen3.5-0.8B 2fc06364715b967f1860aea9cf38778875588b17 5e-6
2b Qwen/Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc 1e-5
g4 google/gemma-4-E2B-it 3e22461f65e89153144f8adb70e3b8c2cc9845a7 2e-5
MODELS

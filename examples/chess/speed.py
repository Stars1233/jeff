"""Milliseconds per move: one position per request (batch size 1), timed on the GPU with nothing else running.

Times the full request (prompt building, tokenising, forward pass, softmax) for the first --count held-out puzzle
positions after --warmup untimed requests. Run from the repository root with `uv run` (needs a CUDA GPU).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import chess
import torch

from jeff.models import load_decision_model
from chessrows import decision_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--positions", type=Path, required=True, help="select_stage1.py output (its test puzzles are timed)")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    model = load_decision_model(checkpoint=args.checkpoint)
    positions = [row for row in map(json.loads, args.positions.open()) if row["split"] == "test"]
    requests = [decision_input(chess.Board(row["fen"])) for row in positions]
    if len(requests) < args.count:
        raise ValueError(f"Only {len(requests)} test positions, {args.count} requested")
    for request in requests[:args.warmup]:
        model.predict([request], batch_size=1)
    timings = []
    for request in requests[:args.count]:
        torch.cuda.synchronize()
        began = time.perf_counter()
        model.predict([request], batch_size=1)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - began) * 1000)
    ordered = sorted(timings)
    print(json.dumps({"checkpoint": args.checkpoint, "requests": len(timings),
                      "median_ms": statistics.median(timings), "mean_ms": statistics.fmean(timings),
                      "p90_ms": ordered[int(0.9 * len(ordered))], "gpu": torch.cuda.get_device_name()}))


if __name__ == "__main__":
    main()

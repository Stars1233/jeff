"""Print the learning rate whose sweep run (runs/<PREFIX><rate>) reached the lowest calibrated dev loss (NLL) at any evaluation.
Usage: best_lr.py PREFIX LR [LR ...]"""

import json
import sys
from pathlib import Path

prefix, rates = sys.argv[1], sys.argv[2:]
if not rates:
    raise SystemExit("Usage: best_lr.py PREFIX LR [LR ...]")
best = {}
for rate in rates:
    path = Path(f"runs/{prefix}{rate}/evaluations.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows:
        raise ValueError(f"{path} has no evaluations")
    best[rate] = min(row["fitted"]["nll"] for row in rows)
    print(f"sweep {rate}: best dev NLL {best[rate]:.4f}", file=sys.stderr)
winner = min(best, key=best.__getitem__)
if winner in (rates[0], rates[-1]):
    print(f"WARNING: best learning rate {winner} is at the edge of the grid", file=sys.stderr)
print(winner)

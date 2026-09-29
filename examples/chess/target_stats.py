"""Describe the soft targets that each candidate softmax temperature would give, to choose the temperature T.

For each T: the quartiles of the best move's probability, the share of positions where the best move gets more than
half, and, for positions where the second-best move is within 0.01 win probability of the best, the median ratio of the
second-best move's weight to the best's (how well near-equal moves share the probability). The release chose T = 0.02.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from chessrows import win_probability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="label.py output")
    parser.add_argument("--temperatures", type=float, nargs="+", required=True)
    parser.add_argument("--limit", type=int, help="Use only the first N labelled positions")
    args = parser.parse_args()
    positions = [json.loads(line) for line in args.labels.open()][:args.limit]
    for temperature in args.temperatures:
        best_share, close_ratio = [], []
        for position in positions:
            wins = sorted((win_probability(m["cp"], m["mate"]) for m in position["moves"]), reverse=True)
            weights = [math.exp((w - wins[0]) / temperature) for w in wins]
            total = sum(weights)
            best_share.append(weights[0] / total)
            if len(wins) > 1 and wins[0] - wins[1] <= 0.01:
                close_ratio.append(weights[1] / weights[0])
        q = statistics.quantiles(best_share, n=4)
        print(json.dumps({"T": temperature, "best_p25": round(q[0], 3), "best_median": round(q[1], 3),
                          "best_p75": round(q[2], 3),
                          "best_over_half": round(sum(s > 0.5 for s in best_share) / len(best_share), 3),
                          "near_equal_positions": len(close_ratio),
                          "near_equal_second_to_best_ratio_median": round(statistics.median(close_ratio), 3)}))


if __name__ == "__main__":
    main()

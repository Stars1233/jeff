"""Estimate a rating from games against Stockfish at known UCI_Elo settings.

Method: maximum likelihood under the Elo model, expected score E = 1 / (1 + 10 ** ((opponent - R) / 400)), with a
win = 1, draw = 0.5 and loss = 0, over all games at every level together. To keep the estimate finite when every
game is lost (or won), one extra virtual draw is added against each opponent level (a standard, stated prior). The
95% interval is where the log-likelihood falls 1.92 below its maximum. The scale is Stockfish's UCI_Elo scale, which
is anchored to CCRL computer ratings, not to Lichess or FIDE ratings.
"""

import argparse
import json
import math
from pathlib import Path


def log_likelihood(rating: float, games: list[tuple[int, float]]) -> float:
    total = 0.0
    for opponent, score in games:
        expected = 1 / (1 + 10 ** ((opponent - rating) / 400))
        total += score * math.log(expected) + (1 - score) * math.log(1 - expected)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    games: list[tuple[int, float]] = []
    levels = []
    for path in args.results:
        summary = json.loads(path.read_text())
        elo = summary["elo"]
        levels.append(elo)
        for game in summary["results"]:
            games.append((elo, {"win": 1.0, "loss": 0.0, "draw": 0.5, "adjudicated draw": 0.5}[game["result"]]))
    with_prior = games + [(level, 0.5) for level in levels]
    grid = [r / 2 for r in range(0, 2 * 3500)]
    values = [log_likelihood(r, with_prior) for r in grid]
    best_index = max(range(len(grid)), key=values.__getitem__)
    inside = [r for r, v in zip(grid, values) if v >= values[best_index] - 1.92]
    print(json.dumps({"games": len(games), "score": sum(s for _, s in games), "estimate": grid[best_index],
                      "ci95": [min(inside), max(inside)], "levels": levels,
                      "method": "maximum likelihood Elo with one virtual draw per level"}))


if __name__ == "__main__":
    main()

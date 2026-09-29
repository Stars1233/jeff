"""Human time per move in Lichess blitz, from the clock comments of a streamed monthly games file (read from stdin).

Keeps "Rated Blitz game" events (no arena games, where berserk removes the increment) with time control 180+0, 180+2, 300+0 or 300+3 that have clock comments, until --games
games are collected. For each player, the time spent on a move is the clock after their previous move minus the clock
after this move, plus the increment. Each player's first move is skipped: Lichess does not run the clock for it.
A negative value can only come from the opponent adding time ("give more time"); such moves are counted and left out.

Writes every per-move time (with the mover's rating, time control and move number) to --output (JSON lines) and a
summary (median, mean, percentiles; overall, by rating band and by time control) to --summary.
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CONTROLS = {"180+0", "180+2", "300+0", "300+3"}
CLOCK = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
BANDS = [("<1200", 0, 1200), ("1200-1599", 1200, 1600), ("1600-1999", 1600, 2000), (">=2000", 2000, 10000)]


def band(rating: int) -> str:
    for name, low, high in BANDS:
        if low <= rating < high:
            return name
    raise ValueError(f"Rating {rating} outside the bands")


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"moves": len(values), "median_s": statistics.median(values), "mean_s": statistics.fmean(values),
            "p10_s": pick(0.1), "p25_s": pick(0.25), "p75_s": pick(0.75), "p90_s": pick(0.9), "p99_s": pick(0.99)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source", required=True, help="URL of the streamed file, recorded in the summary")
    args = parser.parse_args()
    headers: dict[str, str] = {}
    kept = scanned = negative = 0
    times: list[dict] = []
    for line in sys.stdin:
        if line.startswith("["):
            if line.startswith("[Event ") and headers.get("_done"):
                headers = {}
            key, _, value = line[1:].partition(" ")
            headers[key] = value.strip().rstrip("]").strip('"')
            continue
        if not line.strip() or not line.startswith("1."):
            continue
        scanned += 1
        headers["_done"] = "1"
        control = headers.get("TimeControl", "")
        if (control not in CONTROLS or headers.get("Event") != "Rated Blitz game"
                or headers.get("Variant", "Standard") != "Standard"):
            continue
        clocks = [int(h) * 3600 + int(m) * 60 + float(s) for h, m, s in CLOCK.findall(line)]
        if len(clocks) < 4:
            continue
        ratings = (int(headers["WhiteElo"]), int(headers["BlackElo"]))
        increment = int(control.split("+")[1])
        kept += 1
        for index in range(2, len(clocks)):
            spent = clocks[index - 2] - clocks[index] + increment
            if spent < 0:
                negative += 1
                continue
            mover = index % 2
            times.append({"s": round(spent, 2), "rating": ratings[mover], "control": control, "ply": index + 1})
        if kept >= args.games:
            break
    if kept < args.games:
        raise ValueError(f"Only {kept} blitz games with clocks found")
    with args.output.open("w") as stream:
        for row in times:
            stream.write(json.dumps(row) + "\n")
    by_band = defaultdict(list)
    by_control = defaultdict(list)
    for row in times:
        by_band[band(row["rating"])].append(row["s"])
        by_control[row["control"]].append(row["s"])
    summary = {"source": args.source, "games": kept, "games_scanned": scanned, "negative_moves_left_out": negative,
               "overall": describe([row["s"] for row in times]),
               "by_band": {name: describe(by_band[name]) for name, _, _ in BANDS},
               "by_time_control": {name: describe(values) for name, values in sorted(by_control.items())}}
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()

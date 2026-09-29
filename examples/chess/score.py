"""Puzzle accuracy overall and by rating band, from jeff-evaluate predictions on the held-out test file, plus two floors.

Accuracy: the model's top move equals the puzzle's first solution move; also a 95% Wilson interval overall and top-3
accuracy (the solution is among the model's three most likely moves).
--random: the expected accuracy of picking uniformly among the legal moves (the mean of 1 / number of legal moves).
--rule: a no-learning rule that prefers a checkmating move, then a capture that gives check, then any capture, then any
check, then any move, breaking ties uniformly at random (expected accuracy). It shows how much of the puzzle score the
capture (x), check (+) and mate (#) marks in the move notation could explain on their own.
"""

import argparse
import json
import math
from pathlib import Path

BANDS = [("<1000", 0, 1000), ("1000-1499", 1000, 1500), ("1500-1999", 1500, 2000), (">=2000", 2000, 10000)]


def wilson(correct: int, total: int) -> tuple[float, float]:
    z = 1.96
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def band(rating: int) -> str:
    for name, low, high in BANDS:
        if low <= rating < high:
            return name
    raise ValueError(f"Rating {rating} outside the bands")


def tier(san: str) -> int:
    if san.endswith("#"):
        return 0
    if "x" in san and san.endswith("+"):
        return 1
    if "x" in san:
        return 2
    if san.endswith("+"):
        return 3
    return 4


def rule_expected(row: dict) -> float:
    keys = list(row["question"]["criteria"])
    top = min(tier(key) for key in keys)
    chosen = [key for key in keys if tier(key) == top]
    return (row["label"] in chosen) / len(chosen)


def by_band(rows: dict[str, dict], values: dict[str, float], name: str) -> dict:
    result: dict = {"name": name}
    for band_name, _, _ in BANDS:
        members = [key for key, row in rows.items() if band(row["source"]["rating"]) == band_name]
        result[band_name] = sum(values[key] for key in members) / len(members)
    result["overall"] = sum(values.values()) / len(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=Path, required=True, help="test.jsonl from build_rows.py")
    parser.add_argument("--predictions", type=Path, nargs="*", default=[], help="*.predictions.jsonl from jeff-evaluate")
    parser.add_argument("--random", action="store_true", help="Also print the random-choice floor")
    parser.add_argument("--rule", action="store_true", help="Also print the mate/capture/check rule floor")
    args = parser.parse_args()
    if not (args.predictions or args.random or args.rule):
        parser.error("Give --predictions, --random or --rule")
    rows = {row["id"]: row for row in map(json.loads, args.test.open())}
    if args.random:
        print(json.dumps(by_band(rows, {key: 1 / len(row["question"]["criteria"]) for key, row in rows.items()},
                                 "random legal move")))
    if args.rule:
        print(json.dumps(by_band(rows, {key: rule_expected(row) for key, row in rows.items()},
                                 "rule: mate > capture with check > capture > check > any, random within tier")))
    for path in args.predictions:
        predictions = {p["id"]: p for p in map(json.loads, path.open())}
        if set(predictions) != set(rows):
            raise ValueError(f"{path} does not cover the test file exactly")
        correct = {key: float(p["prediction"] == rows[key]["label"]) for key, p in predictions.items()}
        result = by_band(rows, correct, path.name.removesuffix(".predictions.jsonl"))
        result["overall_ci95"] = wilson(int(sum(correct.values())), len(correct))
        top3 = 0
        for key, p in predictions.items():
            order = sorted(range(len(p["options"])), key=lambda i: -p["probabilities"][i])[:3]
            top3 += rows[key]["label"] in [p["options"][i] for i in order]
        result["top3"] = top3 / len(predictions)
        print(json.dumps(result))


if __name__ == "__main__":
    main()

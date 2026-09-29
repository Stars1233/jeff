"""Write the trainer files for one training stage: train.jsonl, dev.jsonl, calibration.jsonl and the held-out puzzle
test.jsonl, every request built by chessrows.py (FEN board, legal moves in SAN as options).

stage1: train, development and calibration rows from the labelled positions (label.py), and the 1,000 held-out test
        puzzles from the positions file (select_stage1.py), labelled with each puzzle's own solution move.
stage2: train rows from the new labelled positions; development, calibration and test are copied unchanged from the
        stage-one folder, so both stages are selected and scored on the same rows.
Both fail if a training row shares a puzzle or game with the held-out files.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from chessrows import test_row, training_row

SPLIT_FILES = {"train": "train.jsonl", "dev": "dev.jsonl", "calibration": "calibration.jsonl"}
HELD_OUT = ("dev.jsonl", "calibration.jsonl", "test.jsonl")
TEST_PUZZLES = 1_000


def write(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def stage1(args: argparse.Namespace) -> dict:
    labelled = [json.loads(line) for line in args.labels.open()]
    by_split: dict[str, list[dict]] = {split: [] for split in SPLIT_FILES}
    for position in labelled:
        if position["split"] not in by_split:
            raise ValueError(f"{position['id']}: unexpected split {position['split']!r} in the labels")
        by_split[position["split"]].append(position)
    tests = [row for row in map(json.loads, args.positions.open()) if row["split"] == "test"]
    if len(tests) != TEST_PUZZLES:
        raise ValueError(f"Expected {TEST_PUZZLES:,} test puzzles, found {len(tests)}")
    test_families = {row["family"] for row in tests}
    summary: dict = {}
    for split, positions in by_split.items():
        if not positions:
            raise ValueError(f"No labelled {split} positions")
        if test_families & {row["family"] for row in positions}:
            raise ValueError(f"The {split} split shares a puzzle with the held-out test")
        write(args.output / SPLIT_FILES[split], [training_row(position, args.temperature) for position in positions])
        summary[split] = {"rows": len(positions), "kinds": Counter(position["kind"] for position in positions)}
    write(args.output / "test.jsonl", [test_row(position) for position in tests])
    summary["test"] = {"rows": len(tests)}
    return summary


def stage2(args: argparse.Namespace) -> dict:
    held: set[str] = set()
    for name in HELD_OUT:
        shutil.copyfile(args.stage_one / name, args.output / name)
        held |= {json.loads(line)["family"] for line in (args.stage_one / name).open()}
    rows = [training_row(json.loads(line), args.temperature) for line in args.labels.open()]
    clash = {row["family"] for row in rows} & held
    if clash:
        raise ValueError(f"{len(clash)} new families overlap the development, calibration or test files")
    write(args.output / "train.jsonl", rows)
    return {"train": {"rows": len(rows), "kinds": Counter(row["family"].split("-", 1)[0] for row in rows)},
            "held_out_from": str(args.stage_one)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    first = stages.add_parser("stage1")
    first.add_argument("--positions", type=Path, required=True, help="select_stage1.py output (for the test puzzles)")
    first.add_argument("--labels", type=Path, required=True, help="label.py output for the same positions")
    second = stages.add_parser("stage2")
    second.add_argument("--labels", type=Path, required=True, help="label.py output for the select_stage2.py positions")
    second.add_argument("--stage-one", type=Path, required=True, help="Stage-one output folder (dev, calibration, test)")
    for sub in (first, second):
        sub.add_argument("--temperature", type=float, required=True, help="Softmax temperature for the soft targets")
        sub.add_argument("--output", type=Path, required=True, help="Folder to create")
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = stage1(args) if args.stage == "stage1" else stage2(args)
    summary.update(stage=args.stage, temperature=args.temperature, labels=str(args.labels))
    (args.output / "build.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

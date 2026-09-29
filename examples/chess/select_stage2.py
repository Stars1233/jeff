"""Stage two: select 300,000 NEW training positions, disjoint from everything used in stage one.

Mix: 30% puzzles in equal numbers per 200-point rating bin, 70% game positions after the first 8 full moves (1 or 2
per game) from games where BOTH players are rated --min-elo or higher. Stage one solved puzzles but lost its games, so
stage two leans towards ordinary positions from stronger players.
The games are streamed from a Lichess month on stdin (the release used 2026-08: curl ... | zstd -dc); the first
--max-games qualifying games in the file are collected and then shuffled. Exclusions:
- puzzles: any puzzle ID in stage one (train, dev, calibration, test);
- games: any game that a stage-one or stage-two puzzle came from (stage one used only 2013-01 games);
- boards: any board (FEN without move counters) that appears anywhere in stage one, or earlier in this selection.
Every new row is in split "train"; the family check against stage one is repeated at the end.
"""

import argparse
import json
import multiprocessing
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import chess

from select_stage1 import TRAIN_BINS, band_of, board_key, game_positions, puzzle_position, read_puzzles


def strong_games(min_elo: int, limit: int) -> list[str]:
    """Collect up to `limit` standard games from stdin in which both players are rated min_elo or higher."""
    chunks: list[str] = []
    lines: list[str] = []
    in_moves = False
    for line in sys.stdin:
        if line.startswith("[Event ") and in_moves:
            text = "".join(lines)
            if qualifies(text, min_elo):
                chunks.append(text)
                if len(chunks) >= limit:
                    return chunks
            lines, in_moves = [], False
        lines.append(line)
        if line.startswith("1.") or line.startswith("0-1") or line.startswith("1-0") or line.startswith("1/2"):
            in_moves = True
    raise ValueError(f"The stream ended after {len(chunks)} qualifying games; {limit} wanted")


def qualifies(text: str, min_elo: int) -> bool:
    headers = dict(re.findall(r'^\[(\w+) "([^"]*)"\]', text, flags=re.M))
    if headers.get("Variant", "Standard") != "Standard" or not headers.get("Event", "").startswith("Rated"):
        return False
    elos = (headers.get("WhiteElo", "?"), headers.get("BlackElo", "?"))
    return all(value.isdigit() and int(value) >= min_elo for value in elos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzles", type=Path, required=True, help="lichess_db_puzzle.csv.zst")
    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--max-games", type=int, default=190_000, help="Qualifying games to collect from stdin")
    parser.add_argument("--stage-one", type=Path, required=True, help="Stage-one positions file (select_stage1.py output)")
    parser.add_argument("--output", type=Path, required=True, help="Positions file to write (JSON lines)")
    parser.add_argument("--train", type=int, default=300_000)
    parser.add_argument("--puzzle-share", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Processes that parse games (does not change the result)")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    rng = random.Random(args.seed)
    old = [json.loads(line) for line in args.stage_one.open()]
    seen = {row["key"] for row in old}
    old_puzzles = {row["family"] for row in old if row["kind"] == "puzzle"}
    old_games = {row["family"] for row in old if row["kind"] == "game"}
    out: list[dict] = []

    puzzles = read_puzzles(args.puzzles)
    source_games = {row["GameUrl"].split("lichess.org/", 1)[1].split("/", 1)[0].split("#", 1)[0]
                    for row in puzzles if row["PuzzleId"] in old_puzzles}
    rng.shuffle(puzzles)
    wanted = round(args.train * args.puzzle_share)
    quotas = [wanted // len(TRAIN_BINS) + (1 if index < wanted % len(TRAIN_BINS) else 0) for index in range(len(TRAIN_BINS))]
    filled = Counter()
    for row in puzzles:
        if row["PuzzleId"] in old_puzzles:
            continue
        band = band_of(int(row["Rating"]), TRAIN_BINS)
        if filled[band] >= quotas[band]:
            continue
        position = puzzle_position(row)
        if position["key"] in seen:
            continue
        seen.add(position["key"])
        filled[band] += 1
        out.append({"split": "train", "kind": "puzzle", "family": row["PuzzleId"], "rating": int(row["Rating"]),
                    "themes": row["Themes"], **position})
        source_games.add(row["GameUrl"].split("lichess.org/", 1)[1].split("/", 1)[0].split("#", 1)[0])
        if sum(filled.values()) == wanted:
            break
    if sum(filled.values()) != wanted:
        raise ValueError(f"Could not fill the puzzle bins: {dict(filled)}")
    print(f"{wanted} new puzzles", flush=True)

    needed = args.train - wanted
    chunks = strong_games(args.min_elo, args.max_games)
    rng.shuffle(chunks)
    print(f"{len(chunks)} games with both players rated {args.min_elo}+", flush=True)
    games = 0
    with multiprocessing.Pool(args.workers) as pool:
        for result in pool.imap(game_positions, [(chunk, args.seed + index) for index, chunk in enumerate(chunks)], chunksize=64):
            if games >= needed:
                break
            if result is None or result["game_id"] in old_games or result["game_id"] in source_games:
                continue
            for fen in result["fens"]:
                board = chess.Board(fen)
                key = board_key(board)
                if key in seen or games >= needed:
                    continue
                seen.add(key)
                games += 1
                out.append({"split": "train", "kind": "game", "family": result["game_id"], "fen": fen, "key": key,
                            "legal_moves": board.legal_moves.count(), "white_elo": result["white_elo"],
                            "black_elo": result["black_elo"], "time_control": result["time_control"]})
    if games != needed:
        raise ValueError(f"Only {games} game positions, {needed} needed")
    old_families = {(row["kind"], row["family"]) for row in old}
    overlap = {(row["kind"], row["family"]) for row in out} & old_families
    if overlap:
        raise ValueError(f"{len(overlap)} families overlap stage one, for example {sorted(overlap)[:3]}")
    with args.output.open("w") as stream:
        for index, row in enumerate(out):
            row["id"] = f"chess-r2-train-{row['kind']}-{index:06d}"
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(Counter(row["kind"] for row in out)), flush=True)


if __name__ == "__main__":
    main()

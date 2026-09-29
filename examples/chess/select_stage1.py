"""Stage one: select chess positions for training, development, calibration and the held-out puzzle test.

Reads the Lichess puzzle database and one month of Lichess standard games (the release used 2013-01), and writes one
line per position with its split, family (puzzle or game ID), FEN and source metadata. Nothing here calls Stockfish.

Mix: the held-out test is 1,000 puzzles, 250 per rating band (<1000, 1000-1499, 1500-1999, >=2000), selected first.
Training, development and calibration are --puzzle-share puzzles (equal numbers per 200-point rating bin) and the rest
game positions. The release used 300,000 training rows at a 60/40 puzzle/game mix, and 2,000 each for development and
calibration.

Puzzles: the CSV FEN is the position before the opponent's move; the first move in Moves is played to reach the
position to solve, and the second move is the correct answer.
Games: skip the first 8 full moves; take one or two positions per game.
Splits are by family: a puzzle or game appears in exactly one split. Positions whose board (FEN without move
counters) repeats an earlier selected position are skipped, so the same board never appears in two splits.
"""

import argparse
import csv
import io
import json
import multiprocessing
import os
import random
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import zstandard

TRAIN_BINS = [(0, 800), (800, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800), (1800, 2000),
              (2000, 2200), (2200, 2400), (2400, 2600), (2600, 10000)]
TEST_BANDS = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 10000)]
MAX_RATING_DEVIATION = 90  # only puzzles whose rating is well established
SKIP_FULL_MOVES = 8


def board_key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def read_puzzles(path: Path) -> list[dict]:
    with path.open("rb") as raw:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
        rows = [row for row in csv.DictReader(text) if int(row["RatingDeviation"]) <= MAX_RATING_DEVIATION]
    return rows


def puzzle_position(row: dict) -> dict:
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    if len(moves) < 2:
        raise ValueError(f"Puzzle {row['PuzzleId']} has fewer than two moves")
    opponent = chess.Move.from_uci(moves[0])
    if opponent not in board.legal_moves:
        raise ValueError(f"Puzzle {row['PuzzleId']}: opponent move {moves[0]} is illegal")
    board.push(opponent)
    answer = chess.Move.from_uci(moves[1])
    if answer not in board.legal_moves:
        raise ValueError(f"Puzzle {row['PuzzleId']}: solution move {moves[1]} is illegal")
    return {"fen": board.fen(), "key": board_key(board), "answer_uci": moves[1], "answer_san": board.san(answer),
            "legal_moves": board.legal_moves.count()}


def split_games(path: Path) -> list[str]:
    with path.open("rb") as raw:
        text = zstandard.ZstdDecompressor().stream_reader(raw).read().decode("utf-8")
    chunks = text.split("\n\n[Event ")
    return [chunks[0]] + ["[Event " + chunk for chunk in chunks[1:]]


def game_positions(arguments: tuple[str, int]) -> dict | None:
    """Return a game's ID and up to two sampled positions after the opening, or None when the game is too short."""
    chunk, seed = arguments
    game = chess.pgn.read_game(io.StringIO(chunk))
    if game is None:
        raise ValueError("Unreadable game chunk")
    if game.errors:
        raise ValueError(f"PGN errors in {game.headers.get('Site')}: {game.errors}")
    site = game.headers["Site"]
    game_id = site.rsplit("/", 1)[1]
    board = game.board()
    candidates = []
    for move in game.mainline_moves():
        board.push(move)
        if board.fullmove_number > SKIP_FULL_MOVES and not board.is_game_over() and board.legal_moves.count() >= 2:
            candidates.append(board.fen())
    if not candidates:
        return None
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(len(candidates), rng.choice([1, 2])))
    return {"game_id": game_id, "white_elo": game.headers.get("WhiteElo"), "black_elo": game.headers.get("BlackElo"),
            "time_control": game.headers.get("TimeControl"), "fens": chosen}


def band_of(rating: int, bands: list[tuple[int, int]]) -> int:
    for index, (low, high) in enumerate(bands):
        if low <= rating < high:
            return index
    raise ValueError(f"Rating {rating} is outside every band")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzles", type=Path, required=True, help="lichess_db_puzzle.csv.zst")
    parser.add_argument("--games", type=Path, required=True, help="A monthly Lichess standard games file (.pgn.zst)")
    parser.add_argument("--output", type=Path, required=True, help="Positions file to write (JSON lines)")
    parser.add_argument("--train", type=int, default=300_000)
    parser.add_argument("--development", type=int, default=2_000)
    parser.add_argument("--calibration", type=int, default=2_000)
    parser.add_argument("--test", type=int, default=1_000)
    parser.add_argument("--puzzle-share", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Processes that parse games (does not change the result)")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    rng = random.Random(args.seed)
    seen: set[str] = set()
    out: list[dict] = []

    puzzles = read_puzzles(args.puzzles)
    rng.shuffle(puzzles)
    print(f"{len(puzzles)} puzzles with rating deviation <= {MAX_RATING_DEVIATION}", flush=True)

    # Held-out test first: 1,000 puzzles, equal numbers per test band.
    per_band = args.test // len(TEST_BANDS)
    if per_band * len(TEST_BANDS) != args.test:
        raise ValueError("--test must divide evenly by the number of test bands")
    band_counts = Counter()
    used_ids: set[str] = set()
    test_games: set[str] = set()  # games the test puzzles came from; never used for game positions
    for row in puzzles:
        band = band_of(int(row["Rating"]), TEST_BANDS)
        if band_counts[band] >= per_band:
            continue
        position = puzzle_position(row)
        if position["key"] in seen:
            continue
        seen.add(position["key"])
        used_ids.add(row["PuzzleId"])
        test_games.add(row["GameUrl"].split("lichess.org/", 1)[1].split("/", 1)[0].split("#", 1)[0])
        band_counts[band] += 1
        out.append({"split": "test", "kind": "puzzle", "family": row["PuzzleId"], "rating": int(row["Rating"]),
                    "themes": row["Themes"], **position})
        if sum(band_counts.values()) == args.test:
            break
    if sum(band_counts.values()) != args.test:
        raise ValueError(f"Could not fill the test bands: {dict(band_counts)}")

    # Training, development and calibration puzzles: equal numbers per 200-point rating bin.
    split_sizes = {"train": args.train, "dev": args.development, "calibration": args.calibration}
    for split, total in split_sizes.items():
        wanted = round(total * args.puzzle_share)
        quotas = [wanted // len(TRAIN_BINS) + (1 if index < wanted % len(TRAIN_BINS) else 0) for index in range(len(TRAIN_BINS))]
        filled = Counter()
        for row in puzzles:
            if row["PuzzleId"] in used_ids:
                continue
            band = band_of(int(row["Rating"]), TRAIN_BINS)
            if filled[band] >= quotas[band]:
                continue
            position = puzzle_position(row)
            if position["key"] in seen:
                continue
            seen.add(position["key"])
            used_ids.add(row["PuzzleId"])
            filled[band] += 1
            out.append({"split": split, "kind": "puzzle", "family": row["PuzzleId"], "rating": int(row["Rating"]),
                        "themes": row["Themes"], **position})
            if sum(filled.values()) == wanted:
                break
        if sum(filled.values()) != wanted:
            raise ValueError(f"Could not fill the {split} puzzle bins: {dict(filled)}")
        print(f"{split}: {wanted} puzzles", flush=True)

    # Game positions.
    chunks = split_games(args.games)
    rng.shuffle(chunks)
    print(f"{len(chunks)} games", flush=True)
    needed = {split: total - round(total * args.puzzle_share) for split, total in split_sizes.items()}
    order = ["dev", "calibration", "train"]
    current = 0
    filled_games = Counter()
    with multiprocessing.Pool(args.workers) as pool:
        for result in pool.imap(game_positions, [(chunk, args.seed + index) for index, chunk in enumerate(chunks)], chunksize=64):
            if current == len(order):
                break
            if result is None or result["game_id"] in test_games:
                continue
            split = order[current]
            for fen in result["fens"]:
                board = chess.Board(fen)
                key = board_key(board)
                if key in seen or filled_games[split] >= needed[split]:
                    continue
                seen.add(key)
                filled_games[split] += 1
                out.append({"split": split, "kind": "game", "family": result["game_id"], "fen": fen, "key": key,
                            "legal_moves": board.legal_moves.count(), "white_elo": result["white_elo"],
                            "black_elo": result["black_elo"], "time_control": result["time_control"]})
            if filled_games[split] >= needed[split]:
                current += 1
    if any(filled_games[split] != needed[split] for split in order):
        raise ValueError(f"Not enough game positions: {dict(filled_games)} of {needed}")

    families: dict[str, str] = {}
    for row in out:
        previous = families.setdefault(row["family"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Family {row['family']} appears in {previous} and {row['split']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for index, row in enumerate(out):
            row["id"] = f"chess-{row['split']}-{row['kind']}-{index:06d}"
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(Counter(f"{row['split']}/{row['kind']}" for row in out)), flush=True)


if __name__ == "__main__":
    main()

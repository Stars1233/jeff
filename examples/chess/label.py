"""Score every legal move of each selected position with Stockfish (MultiPV = number of legal moves, fixed depth).

Labels every position in the file except the held-out test puzzles, which are scored against the puzzle's own
solution move instead. The release used Stockfish 19, depth 10, a cap of 4,000,000 nodes per position, one thread and
64 MB hash per engine, with the hash cleared for every position.

Each output line keeps the position fields and adds "moves": a list of {uci, san, cp, mate} from the point of view of
the side to move (cp is null when the score is a forced mate; mate is the signed number of moves to mate), plus the
depth the search actually reached and the nodes it used. The output is written in the order positions finish.
"""

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import chess
import chess.engine

ENGINE: chess.engine.SimpleEngine | None = None
DEPTH = 0
NODE_CAP = 0


def start(engine_path: str, depth: int, hash_mb: int, node_cap: int) -> None:
    global ENGINE, DEPTH, NODE_CAP
    ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    ENGINE.configure({"Threads": 1, "Hash": hash_mb})
    DEPTH = depth
    NODE_CAP = node_cap


def label(row: dict) -> dict:
    if ENGINE is None:
        raise RuntimeError("Engine not started")
    board = chess.Board(row["fen"])
    count = board.legal_moves.count()
    # A fresh game per position (clears the hash) keeps the labels reproducible and independent of the order.
    # The node cap stops the rare positions where a full MultiPV search explodes (one took minutes without reaching
    # depth 10); the depth each position actually reached is recorded.
    infos = ENGINE.analyse(board, chess.engine.Limit(depth=DEPTH, nodes=NODE_CAP), multipv=count, game=object())
    moves = []
    for info in infos:
        move = info["pv"][0]
        score = info["score"].pov(board.turn)
        moves.append({"uci": move.uci(), "san": board.san(move), "cp": score.score(), "mate": score.mate()})
    if sorted(item["uci"] for item in moves) != sorted(move.uci() for move in board.legal_moves):
        raise ValueError(f"{row['id']}: Stockfish returned {len(moves)} lines for {count} legal moves")
    for item in moves:
        if (item["cp"] is None) == (item["mate"] is None):
            raise ValueError(f"{row['id']}: a move has neither or both of a centipawn and a mate score")
    reached = min(info["depth"] for info in infos)
    return {**row, "moves": moves, "stockfish_depth": DEPTH, "stockfish_node_cap": NODE_CAP,
            "depth_reached": reached, "nodes": infos[0]["nodes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True, help="select_stage1.py or select_stage2.py output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", required=True, help="Path to the Stockfish binary")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--node-cap", type=int, required=True, help="Stop a position's search after this many nodes")
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Stockfish processes, one thread each")
    args = parser.parse_args()
    if args.node_cap < 1 or args.depth < 1:
        raise ValueError("--depth and --node-cap must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    chosen = [row for row in map(json.loads, args.positions.open()) if row["split"] != "test"]
    if not chosen:
        raise ValueError(f"{args.positions} has no positions outside the held-out test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    began = time.time()
    done = 0
    with multiprocessing.Pool(args.workers, initializer=start, initargs=(args.engine, args.depth, args.hash_mb, args.node_cap)) as pool, \
            args.output.open("w") as stream:
        # Unordered, so one slow position does not hold back the output of the others.
        for result in pool.imap_unordered(label, chosen, chunksize=4):
            stream.write(json.dumps(result) + "\n")
            done += 1
            if done % 2000 == 0 or done == len(chosen):
                elapsed = time.time() - began
                print(json.dumps({"done": done, "of": len(chosen), "elapsed_s": round(elapsed, 1),
                                  "eta_s": round(elapsed / done * (len(chosen) - done), 1)}), flush=True)
    print(json.dumps({"rows": done, "depth": args.depth, "node_cap": args.node_cap, "workers": args.workers,
                      "seconds": round(time.time() - began, 1)}), flush=True)


if __name__ == "__main__":
    main()

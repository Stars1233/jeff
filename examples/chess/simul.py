"""Many simultaneous blitz-paced games: Jeff (one batched move server on the GPU) against stand-in human opponents.

Each opponent is Stockfish 19 at UCI_LimitStrength / UCI_Elo 1350 (one thread, --move-time seconds per search, from a
shared pool of --engines processes), whose move is released only after a think time sampled from real Lichess blitz
move times (human_times.py output). If the search takes longer than the sampled time, the move is released
when the search ends. Jeff answers every waiting board in one batch (move_server.MoveServer).

Every move is logged with timestamps (seconds since the start) to --log (JSON lines), as are game starts and ends.
Games end by the normal rules (checkmate, stalemate, insufficient material, threefold repetition, fifty moves, all
claimed automatically), by adjudication as a draw after --max-plies half-moves, or are marked unfinished when the
wall-clock cap is reached. Run from the repository root with `uv run` (needs a CUDA GPU).
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import chess
import chess.engine

from move_server import MoveServer


class Log:
    def __init__(self, path: Path, start: float) -> None:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        self.stream = path.open("w")
        self.start = start

    def write(self, **event) -> None:
        event["t"] = round(time.monotonic() - self.start, 4)
        self.stream.write(json.dumps(event) + "\n")
        self.stream.flush()


async def play(game: int, jeff_colour: chess.Color, server: MoveServer, engines: asyncio.Queue, think: list[float],
               rng: random.Random, args: argparse.Namespace, log: Log, results: dict) -> None:
    board = chess.Board()
    log.write(kind="start", game=game, jeff="white" if jeff_colour == chess.WHITE else "black")
    while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
        if board.turn == jeff_colour:
            move, latency, batch = await server.move(board)
            log.write(kind="move", game=game, ply=board.ply() + 1, by="jeff", uci=move.uci(), san=board.san(move),
                      latency_ms=round(latency * 1000, 2), batch=batch)
        else:
            sampled = rng.choice(think)
            began = time.monotonic()
            engine = await engines.get()
            try:
                result = await engine.play(board, chess.engine.Limit(time=args.move_time), game=game)
            finally:
                engines.put_nowait(engine)
            if result.move is None:
                raise RuntimeError(f"Stockfish returned no move in game {game}")
            computed = time.monotonic() - began
            if sampled > computed:
                await asyncio.sleep(sampled - computed)
            move = result.move
            log.write(kind="move", game=game, ply=board.ply() + 1, by="opponent", uci=move.uci(), san=board.san(move),
                      think_sampled_s=sampled, search_s=round(computed, 3))
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result_text, reason = "draw", f"adjudicated at {args.max_plies} plies"
    elif outcome.winner is None:
        result_text, reason = "draw", outcome.termination.name
    else:
        result_text, reason = ("win" if outcome.winner == jeff_colour else "loss"), outcome.termination.name
    results[game] = result_text
    log.write(kind="end", game=game, result=result_text, reason=reason, plies=board.ply())


async def main_async(args: argparse.Namespace) -> None:
    think = [json.loads(line)["s"] for line in args.human_times.open()]
    server = MoveServer(args.checkpoint, max_batch=args.max_batch)
    server.warm_up()
    engines: asyncio.Queue = asyncio.Queue()
    for _ in range(args.engines):
        _, engine = await chess.engine.popen_uci(args.engine)
        await engine.configure({"Threads": 1, "Hash": 16, "UCI_LimitStrength": True, "UCI_Elo": args.elo})
        engines.put_nowait(engine)
    start = time.monotonic()
    log = Log(args.log, start)
    log.write(kind="config", games=args.games, engines=args.engines, elo=args.elo, move_time_s=args.move_time,
              max_batch=args.max_batch, cap_s=args.cap, checkpoint=args.checkpoint)
    worker = asyncio.create_task(server.run())
    results: dict[int, str] = {}
    tasks = [asyncio.create_task(play(game, chess.WHITE if game % 2 == 0 else chess.BLACK, server, engines, think,
                                      random.Random(args.seed + game), args, log, results))
             for game in range(args.games)]
    done, pending = await asyncio.wait(tasks, timeout=args.cap)
    for task in done:
        task.result()  # re-raise any error from a game
    for task in pending:
        task.cancel()
    # Let the cancelled games return their Stockfish engines to the pool; anything but a cancellation is an error.
    for outcome in await asyncio.gather(*pending, return_exceptions=True):
        if not isinstance(outcome, asyncio.CancelledError):
            raise RuntimeError(f"A game stopped with an unexpected result: {outcome!r}")
    unfinished = len(pending)
    log.write(kind="stop", finished=len(done), unfinished=unfinished)
    worker.cancel()
    while not engines.empty():
        await engines.get_nowait().quit()
    moves = [json.loads(line) for line in args.log.open()]
    latencies = sorted(event["latency_ms"] for event in moves if event.get("by") == "jeff")
    summary = {"games": args.games, "finished": len(done), "unfinished": unfinished,
               "wins": sum(v == "win" for v in results.values()), "draws": sum(v == "draw" for v in results.values()),
               "losses": sum(v == "loss" for v in results.values()), "jeff_moves": len(latencies),
               "jeff_latency_median_ms": statistics.median(latencies),
               "jeff_latency_p95_ms": latencies[int(0.95 * len(latencies))], "jeff_latency_max_ms": latencies[-1],
               "batch_sizes": {"median": statistics.median(b["size"] for b in server.batches),
                               "max": max(b["size"] for b in server.batches), "batches": len(server.batches)},
               "wall_s": time.monotonic() - start, "engines": args.engines, "move_time_s": args.move_time, "elo": args.elo}
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--engine", required=True, help="Path to the Stockfish binary")
    parser.add_argument("--human-times", type=Path, required=True, help="human_times.py --output")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--engines", type=int, default=24)
    parser.add_argument("--elo", type=int, default=1350)
    parser.add_argument("--move-time", type=float, default=0.2)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--cap", type=float, default=900, help="Wall-clock cap in seconds")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

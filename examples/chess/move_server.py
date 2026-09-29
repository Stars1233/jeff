"""A batched Jeff chess move server, and a throughput benchmark for it.

MoveServer loads a chess checkpoint with jeff.models.load_decision_model. Callers `await server.move(board)`;
a single worker takes every board that is waiting (up to --max-batch), builds the requests with
chessrows.decision_input(board), runs them through the model in one batch, and answers each with its
highest-probability legal move. The forward pass runs in a worker thread so the event loop keeps serving games.

`benchmark` mode measures latency (one batch, from request building to answers) and moves per second at fixed batch
sizes on real game positions. Run from the repository root with `uv run` (needs a CUDA GPU).
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import torch

from jeff.models import load_decision_model
from chessrows import decision_input


def choose(model, boards: list[chess.Board]) -> list[chess.Move]:
    requests = [decision_input(board) for board in boards]
    distributions = model.predict(requests, batch_size=len(requests))
    moves = []
    for board, request, probabilities in zip(boards, requests, distributions, strict=True):
        keys = list(request["question"]["criteria"])
        moves.append(board.parse_san(keys[max(range(len(keys)), key=probabilities.__getitem__)]))
    return moves


@dataclass
class Pending:
    board: chess.Board
    future: asyncio.Future
    queued: float


@dataclass
class MoveServer:
    checkpoint: str
    max_batch: int = 128
    batches: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = load_decision_model(checkpoint=self.checkpoint)
        self.queue: asyncio.Queue[Pending] = asyncio.Queue()

    def warm_up(self) -> None:
        """Run a few untimed batches first, so the first real requests do not pay for GPU start-up."""
        for size in (1, 2, 4, 8, 16, 32, 64, 128):
            choose(self.model, [chess.Board() for _ in range(min(size, self.max_batch))])

    async def move(self, board: chess.Board) -> tuple[chess.Move, float, int]:
        """Returns the move, the time from request to answer in seconds, and the size of the batch it went in."""
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(Pending(board.copy(stack=False), future, time.monotonic()))
        return await future

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            waiting = [await self.queue.get()]
            while not self.queue.empty() and len(waiting) < self.max_batch:
                waiting.append(self.queue.get_nowait())
            began = time.monotonic()
            moves = await loop.run_in_executor(None, choose, self.model, [item.board for item in waiting])
            done = time.monotonic()
            self.batches.append({"t": done, "size": len(waiting), "seconds": done - began})
            for item, move in zip(waiting, moves, strict=True):
                item.future.set_result((move, done - item.queued, len(waiting)))


def benchmark(args: argparse.Namespace) -> None:
    model = load_decision_model(checkpoint=args.checkpoint)
    boards = [chess.Board(row["fen"]) for row in map(json.loads, args.positions.open()) if row["kind"] == "game"]
    if len(boards) < max(args.sizes) * (args.repeats + 1):
        raise ValueError("Not enough game positions for the benchmark")
    results = []
    offset = 0
    for size in args.sizes:
        choose(model, boards[offset:offset + size])  # warm-up at this size
        offset += size
        latencies = []
        for _ in range(args.repeats):
            batch = boards[offset:offset + size]
            offset += size
            torch.cuda.synchronize()
            began = time.perf_counter()
            choose(model, batch)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - began)
        ordered = sorted(latencies)
        median = statistics.median(latencies)
        result = {"batch_size": size, "repeats": args.repeats, "latency_median_ms": median * 1000,
                  "latency_p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000,
                  "moves_per_second": size / median}
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps({"checkpoint": args.checkpoint, "gpu": torch.cuda.get_device_name(),
                                       "positions": str(args.positions), "results": results}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["benchmark"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--positions", type=Path, required=True, help="Positions file (the release used select_stage2.py output); rows of kind 'game' are used")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()

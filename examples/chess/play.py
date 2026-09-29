"""Play games between a Jeff chess checkpoint and Stockfish at a limited strength, and time the model's moves.

The model plays its top move each turn with no search: the request is chessrows.decision_input(board)
and the move is the option with the highest probability. Only legal moves are offered, so an illegal move cannot
happen. Stockfish plays with UCI_LimitStrength and UCI_Elo, one thread, a fixed time per move.

Run from the repository root with `uv run python examples/chess/play.py ...` (needs a CUDA GPU).
A game ends by checkmate, stalemate, insufficient material, threefold repetition or the fifty-move rule (claimed
automatically); a game still running after --max-plies half-moves is adjudicated a draw and counted separately.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

from jeff.models import load_decision_model
from chessrows import decision_input


def model_move(model, board: chess.Board) -> tuple[chess.Move, float]:
    request = decision_input(board)
    keys = list(request["question"]["criteria"])
    torch.cuda.synchronize()
    began = time.perf_counter()
    probabilities = model.predict([request], batch_size=1)[0]
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - began) * 1000
    best = keys[max(range(len(keys)), key=probabilities.__getitem__)]
    return board.parse_san(best), elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--engine", help="Path to the Stockfish binary (Stockfish opponent only)")
    parser.add_argument("--elo", type=int, help="Stockfish UCI_Elo (omit with --opponent random)")
    parser.add_argument("--opponent", choices=["stockfish", "random"], default="stockfish",
                        help="random: a player that picks a uniformly random legal move (a sanity check, not a rating anchor)")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--move-time", type=float, default=0.2)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    model = load_decision_model(checkpoint=args.checkpoint)
    if (args.opponent == "stockfish") != (args.elo is not None):
        raise ValueError("--elo is required for a Stockfish opponent and not allowed for the random opponent")
    if (args.opponent == "stockfish") != (args.engine is not None):
        raise ValueError("--engine is required for a Stockfish opponent and not allowed for the random opponent")
    rng = random.Random(args.seed)
    opponent_name = f"Stockfish UCI_Elo {args.elo}" if args.opponent == "stockfish" else "random mover"
    engine = None
    if args.opponent == "stockfish":
        engine = chess.engine.SimpleEngine.popen_uci(args.engine)
        engine.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": True, "UCI_Elo": args.elo})
    timings: list[float] = []
    results = []
    pgns = []
    for game_index in range(args.games):
        model_colour = chess.WHITE if game_index % 2 == 0 else chess.BLACK
        board = chess.Board()
        engine_game = object()
        while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
            if board.turn == model_colour:
                move, elapsed = model_move(model, board)
                timings.append(elapsed)
            elif engine is None:
                move = rng.choice(list(board.legal_moves))
            else:
                move = engine.play(board, chess.engine.Limit(time=args.move_time), game=engine_game).move
                if move is None:
                    raise RuntimeError("Stockfish returned no move")
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            result, reason = "adjudicated draw", f"reached {args.max_plies} plies"
        elif outcome.winner is None:
            result, reason = "draw", outcome.termination.name
        else:
            result, reason = ("win" if outcome.winner == model_colour else "loss"), outcome.termination.name
        game = chess.pgn.Game.from_board(board)
        game.headers.update({"White": "Jeff chess" if model_colour == chess.WHITE else opponent_name,
                             "Black": "Jeff chess" if model_colour == chess.BLACK else opponent_name,
                             "Event": f"Jeff chess vs {opponent_name}", "Round": str(game_index + 1)})
        pgns.append(str(game))
        record = {"game": game_index + 1, "model_colour": "white" if model_colour == chess.WHITE else "black",
                  "result": result, "reason": reason, "plies": board.ply()}
        results.append(record)
        print(json.dumps(record), flush=True)
    if engine is not None:
        engine.quit()
    summary = {"elo": args.elo, "opponent": opponent_name, "games": len(results), "move_time_s": args.move_time,
               "wins": sum(r["result"] == "win" for r in results),
               "draws": sum(r["result"] in ("draw", "adjudicated draw") for r in results),
               "adjudicated_draws": sum(r["result"] == "adjudicated draw" for r in results),
               "losses": sum(r["result"] == "loss" for r in results),
               "model_move_ms_median": statistics.median(timings), "model_move_ms_mean": statistics.fmean(timings),
               "model_moves": len(timings), "checkpoint": args.checkpoint,
               "results": results}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    args.output.with_suffix(".pgn").write_text("\n\n".join(pgns) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()

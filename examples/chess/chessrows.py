"""Build Jeff decision rows for chess positions. This is the single place where a position becomes a model request.

decision_input(board) returns the "state" and "question" the model reads: the position in Forsyth-Edwards Notation
(FEN) and a choice question whose options are the legal moves in standard algebraic notation (SAN), with no
description, sorted by character code. Every caller (training, evaluation, games, a web app) must build exactly this
request for a position; PROMPT_FORMAT.md describes it field by field.

training_row(...) adds the id, family, soft target, hard label and source fields used by the trainer, and test_row(...)
builds a held-out puzzle row whose label is the puzzle's own solution move.
"""

import math
from collections.abc import Sequence

import chess

LICHESS_WIN_SLOPE = 0.00368208  # Lichess's centipawn-to-win-chance slope
MAX_OPTIONS = 255  # the most options a Jeff question may have; a chess position has at most 218 legal moves


def colour(turn: chess.Color) -> str:
    return "white" if turn == chess.WHITE else "black"


def state(board: chess.Board) -> dict[str, str]:
    # python-chess writes an en passant square only when an en passant capture is legal.
    return {"fen": board.fen()}


def instructions(board: chess.Board) -> str:
    return (f"You are playing {colour(board.turn)}. Choose the best move in this position. "
            "The position is given in Forsyth-Edwards Notation (FEN).")


def option_keys(board: chess.Board) -> list[str]:
    keys = sorted(board.san(move) for move in board.legal_moves)
    if not keys:
        raise ValueError("The position has no legal moves")
    if len(keys) > MAX_OPTIONS:
        raise ValueError(f"{len(keys)} legal moves exceed the {MAX_OPTIONS}-option limit")
    return keys


def decision_input(board: chess.Board) -> dict:
    """The request the model reads: state and a choice question whose options are the legal moves (SAN, sorted)."""
    return {"state": state(board),
            "question": {"type": "choice", "instructions": instructions(board),
                         "criteria": {key: None for key in option_keys(board)}}}


def win_probability(cp: int | None, mate: int | None) -> float:
    """Lichess's conversion from a centipawn score to a win chance; a forced mate is a certain win or loss."""
    if mate is not None:
        if mate == 0:
            raise ValueError("A mate score of zero has no side")
        return 1.0 if mate > 0 else 0.0
    if cp is None:
        raise ValueError("A move needs a centipawn or mate score")
    return 1.0 / (1.0 + math.exp(-LICHESS_WIN_SLOPE * cp))


def soft_target(keys: Sequence[str], moves: Sequence[dict], temperature: float) -> list[float]:
    """Softmax over the moves' win probabilities at the given temperature, in option order."""
    wins = {move["san"]: win_probability(move["cp"], move["mate"]) for move in moves}
    if set(wins) != set(keys):
        raise ValueError("Stockfish moves differ from the legal moves")
    best = max(wins.values())
    weights = [math.exp((wins[key] - best) / temperature) for key in keys]
    total = sum(weights)
    return [weight / total for weight in weights]


def training_row(position: dict, temperature: float) -> dict:
    """A trainer row from a labelled position (see label.py): soft target from Stockfish, label = Stockfish's best."""
    board = chess.Board(position["fen"])
    request = decision_input(board)
    keys = list(request["question"]["criteria"])
    target = soft_target(keys, position["moves"], temperature)
    label = keys[max(range(len(keys)), key=target.__getitem__)]
    return {"id": position["id"], "suite": "chess", "family": f"{position['kind']}-{position['family']}", **request,
            "target": target, "label": label, "source": source(position) | {"stockfish_depth": position["stockfish_depth"],
                                                                           "softmax_temperature": temperature}}


def test_row(position: dict) -> dict:
    """A held-out puzzle row: the label is the puzzle's own solution move."""
    board = chess.Board(position["fen"])
    request = decision_input(board)
    if position["answer_san"] not in request["question"]["criteria"]:
        raise ValueError(f"{position['id']}: the solution move is not among the options")
    return {"id": position["id"], "suite": "chess", "family": f"{position['kind']}-{position['family']}", **request,
            "target": position["answer_san"], "label": position["answer_san"], "source": source(position)}


def source(position: dict) -> dict:
    if position["kind"] == "puzzle":
        return {"dataset": "lichess-puzzles", "upstream_id": position["family"], "rating": position["rating"],
                "themes": position["themes"], "license": "CC0"}
    # Game rows of both stages carry this dataset name (stage two's games are from a later month). It is kept as it was
    # so the rebuilt training files match the released run exactly; the model never sees the source field.
    return {"dataset": "lichess-games-2013-01", "upstream_id": position["family"], "license": "CC0"}

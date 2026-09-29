"""The chess example (examples/chess): requests must match PROMPT_FORMAT.md exactly, and the Stockfish-score maths
must give the documented targets. Needs python-chess (`uv sync --extra chess`); skipped without it."""

import importlib.util
import json
import math
import re
from pathlib import Path
from types import ModuleType

import pytest

chess = pytest.importorskip("chess")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "chess"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chessrows = load("chessrows")


def documented_examples() -> list[tuple[str, dict, dict]]:
    """(FEN, request, full training row) for each worked example in PROMPT_FORMAT.md."""
    text = (EXAMPLE / "PROMPT_FORMAT.md").read_text()
    sections = re.split(r"^## Worked example \d \(\w+\): ", text, flags=re.M)[1:]
    examples = []
    for section in sections:
        fen = section.splitlines()[0].strip()
        blocks = [json.loads(block) for block in re.findall(r"```json\n(.*?)```", section, flags=re.S)]
        assert len(blocks) == 2, "each worked example has a request and a full training row"
        examples.append((fen, blocks[0], blocks[1]))
    assert len(examples) == 2
    return examples


@pytest.mark.parametrize("fen,request_json,row", documented_examples())
def test_requests_match_the_worked_examples_exactly(fen: str, request_json: dict, row: dict) -> None:
    board = chess.Board(fen)
    built = chessrows.decision_input(board)
    assert built == request_json
    # Same keys in the same order, and the same bytes when serialised.
    assert list(built["question"]["criteria"]) == list(request_json["question"]["criteria"])
    assert json.dumps(built, ensure_ascii=False) == json.dumps(request_json, ensure_ascii=False)
    assert {key: row[key] for key in ("state", "question")} == request_json
    keys = list(row["question"]["criteria"])
    assert row["label"] == keys[max(range(len(keys)), key=row["target"].__getitem__)]


def test_worked_example_instructions_and_option_order() -> None:
    (start_fen, start, _), (middle_fen, middle, _) = documented_examples()
    assert start_fen == chess.STARTING_FEN
    assert start["question"]["instructions"] == ("You are playing white. Choose the best move in this position. "
                                                 "The position is given in Forsyth-Edwards Notation (FEN).")
    assert middle["question"]["instructions"] == ("You are playing black. Choose the best move in this position. "
                                                  "The position is given in Forsyth-Edwards Notation (FEN).")
    assert list(start["question"]["criteria"])[:5] == ["Na3", "Nc3", "Nf3", "Nh3", "a3"]
    keys = list(middle["question"]["criteria"])
    assert keys == sorted(keys) and len(keys) == 29
    assert all(value is None for value in middle["question"]["criteria"].values())
    board = chess.Board()
    for san in "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 Nbd7 Rc1 c6 Bd3 dxc4 Bxc4".split():
        board.push_san(san)
    assert board.fen() == middle_fen


def test_win_probability_follows_the_lichess_formula() -> None:
    assert chessrows.win_probability(0, None) == 0.5
    assert chessrows.win_probability(100, None) == pytest.approx(1 / (1 + math.exp(-0.368208)))
    assert chessrows.win_probability(-100, None) == pytest.approx(1 - chessrows.win_probability(100, None))
    assert chessrows.win_probability(None, 3) == 1.0
    assert chessrows.win_probability(None, -1) == 0.0
    with pytest.raises(ValueError):
        chessrows.win_probability(None, 0)
    with pytest.raises(ValueError):
        chessrows.win_probability(None, None)


def test_soft_target_is_a_softmax_of_win_probabilities() -> None:
    temperature = 0.02
    # Scores chosen so the win probabilities are exactly 0.5 (cp 0), p(+100) and 1 (mate for the mover).
    moves = [{"san": "a3", "cp": 0, "mate": None}, {"san": "Nf3", "cp": 100, "mate": None},
             {"san": "Qh5#", "cp": None, "mate": 1}]
    keys = ["Nf3", "Qh5#", "a3"]
    target = chessrows.soft_target(keys, moves, temperature)
    wins = [chessrows.win_probability(100, None), 1.0, 0.5]
    weights = [math.exp((win - 1.0) / temperature) for win in wins]
    assert target == pytest.approx([weight / sum(weights) for weight in weights])
    assert sum(target) == pytest.approx(1.0)
    # Equal scores share the probability equally.
    close = chessrows.soft_target(["a", "b"], [{"san": "a", "cp": None, "mate": 2}, {"san": "b", "cp": None, "mate": 2}],
                                  temperature)
    assert close == [0.5, 0.5]
    with pytest.raises(ValueError):
        chessrows.soft_target(["Nf3", "a3"], moves, temperature)


def test_a_move_slightly_worse_keeps_e_to_the_minus_one_of_the_weight() -> None:
    better = 0.0
    worse = -math.log(1 / 0.48 - 1) / chessrows.LICHESS_WIN_SLOPE  # centipawns with a win chance of exactly 0.48
    target = chessrows.soft_target(["a", "b"], [{"san": "a", "cp": better, "mate": None},
                                                {"san": "b", "cp": worse, "mate": None}], 0.02)
    assert target[1] / target[0] == pytest.approx(math.exp(-1))


def test_training_and_test_rows() -> None:
    board = chess.Board()
    moves = [{"uci": move.uci(), "san": board.san(move), "cp": 30 if board.san(move) == "e4" else 0, "mate": None}
             for move in board.legal_moves]
    position = {"id": "chess-train-game-000001", "kind": "game", "family": "abcd1234", "fen": board.fen(),
                "moves": moves, "stockfish_depth": 10}
    row = chessrows.training_row(position, 0.02)
    assert row["label"] == "e4" and row["suite"] == "chess" and row["family"] == "game-abcd1234"
    assert row["source"] == {"dataset": "lichess-games-2013-01", "upstream_id": "abcd1234", "license": "CC0",
                             "stockfish_depth": 10, "softmax_temperature": 0.02}
    puzzle = {"id": "chess-test-puzzle-000002", "kind": "puzzle", "family": "QYhT8", "fen": board.fen(),
              "answer_san": "Nf3", "rating": 2168, "themes": "opening"}
    test = chessrows.test_row(puzzle)
    assert test["label"] == test["target"] == "Nf3" and test["source"]["rating"] == 2168
    with pytest.raises(ValueError):
        chessrows.test_row(puzzle | {"answer_san": "Nf6"})


def test_en_passant_square_only_when_the_capture_is_legal() -> None:
    board = chess.Board()
    board.push_san("e4")  # no black pawn can capture en passant
    assert chessrows.decision_input(board)["state"]["fen"].split()[3] == "-"
    for san in ("a6", "e5", "d5"):
        board.push_san(san)
    assert chessrows.decision_input(board)["state"]["fen"].split()[3] == "d6"  # exd6 is legal
    assert "exd6" in chessrows.decision_input(board)["question"]["criteria"]
    # Pinned: after bxc6 the white king on a5 would stand in check from the rook on h5, so the capture is illegal.
    pinned = chess.Board("4k3/2p5/8/KP5r/8/8/8/8 b - - 0 1")
    pinned.push_san("c5")
    assert chessrows.decision_input(pinned)["state"]["fen"] == "4k3/8/8/KPp4r/8/8/8/8 w - - 0 2"
    assert "bxc6" not in chessrows.decision_input(pinned)["question"]["criteria"]


def test_a_position_without_legal_moves_is_refused() -> None:
    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    with pytest.raises(ValueError, match="no legal moves"):
        chessrows.decision_input(mate)

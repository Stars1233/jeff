"""Print the two worked examples in PROMPT_FORMAT.md: the request and the full training row for the start position and
one middlegame position, labelled exactly as the training data was (Stockfish, depth 10 with a 4,000,000-node cap,
MultiPV = all legal moves, softmax at the given temperature)."""

import argparse
import json

import chess
import chess.engine

from chessrows import decision_input, training_row

# Queen's Gambit Declined after 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O 6.Nf3 Nbd7 7.Rc1 c6 8.Bd3 dxc4 9.Bxc4
MIDDLEGAME_MOVES = "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 Nbd7 Rc1 c6 Bd3 dxc4 Bxc4".split()
DEPTH = 10
NODE_CAP = 4_000_000


def middlegame() -> str:
    board = chess.Board()
    for san in MIDDLEGAME_MOVES:
        board.push_san(san)
    return board.fen()


def labelled(fen: str, engine: chess.engine.SimpleEngine) -> dict:
    board = chess.Board(fen)
    infos = engine.analyse(board, chess.engine.Limit(depth=DEPTH, nodes=NODE_CAP), multipv=board.legal_moves.count(),
                           game=object())
    moves = []
    for info in infos:
        score = info["score"].pov(board.turn)
        moves.append({"uci": info["pv"][0].uci(), "san": board.san(info["pv"][0]), "cp": score.score(), "mate": score.mate()})
    return {"id": "example", "kind": "example", "family": "example", "fen": fen, "moves": moves, "stockfish_depth": DEPTH}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="Path to the Stockfish binary")
    parser.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args()
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 64})
    for name, fen in (("start", chess.STARTING_FEN), ("middlegame", middlegame())):
        row = training_row(labelled(fen, engine), args.temperature)
        row["source"] = {"dataset": "worked-example", "stockfish_depth": DEPTH, "softmax_temperature": args.temperature}
        print(f"### {name}: {fen}")
        print("request only (what a caller sends):")
        print(json.dumps(decision_input(chess.Board(fen)), indent=2))
        print("full training row:")
        print(json.dumps(row, indent=2))
    engine.quit()


if __name__ == "__main__":
    main()

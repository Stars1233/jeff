"""Record Jeff's move probabilities for one game of the simultaneous run, for the featured-game video.

Replays the game from the simul.py log through the same checkpoint, one position at a time. For every Jeff move it
writes the ply, the log time, the move played, Jeff's response time in the run, the number of legal moves and the five
most likely moves with their probabilities. Fails if the replayed top move differs from the move Jeff played in the
run. Run from the repository root with `uv run` (needs a CUDA GPU).
"""

import argparse
import json
from pathlib import Path

import chess

from jeff.models import load_decision_model
from chessrows import decision_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="simul.py --log")
    parser.add_argument("--game", type=int, required=True, help="Game number in the log (0-based; the release used 62)")
    parser.add_argument("--checkpoint", required=True, help="The checkpoint that played the run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    events = [json.loads(line) for line in args.log.open()]
    if events[0]["kind"] != "config":
        raise ValueError("The log must start with its config line")
    moves = [event for event in events if event["kind"] == "move" and event["game"] == args.game]
    if not moves:
        raise ValueError(f"Game {args.game} has no moves in {args.log}")
    model = load_decision_model(checkpoint=args.checkpoint)
    board = chess.Board()
    decisions = []
    for event in moves:
        if event["ply"] != board.ply() + 1:
            raise ValueError(f"Game {args.game}: ply {event['ply']} is out of order")
        if event["by"] == "jeff":
            request = decision_input(board)
            keys = list(request["question"]["criteria"])
            probabilities = model.predict([request], batch_size=1)[0]
            ranked = sorted(zip(keys, probabilities, strict=True), key=lambda item: -item[1])
            if ranked[0][0] != event["san"]:
                raise ValueError(f"Game {args.game} ply {event['ply']}: the replay prefers {ranked[0][0]}, "
                                 f"but Jeff played {event['san']} in the run")
            decisions.append({"ply": event["ply"], "t": event["t"], "played": event["san"],
                              "latency_ms": event["latency_ms"], "top": [[san, p] for san, p in ranked[:5]],
                              "legal": len(keys)})
        board.push_uci(event["uci"])
    args.output.write_text(json.dumps(decisions, indent=1) + "\n")
    print(json.dumps({"game": args.game, "jeff_moves": len(decisions), "result": board.result(claim_draw=True)}))


if __name__ == "__main__":
    main()

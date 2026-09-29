# Jeff chess: request format

Jeff-Qwen3.5-0.8B-Chess was trained on rows built by one Python function. Anything that plays against the model (a
web app, a game script) must build **exactly** the same request for a position, or the model sees text it was never
trained on.

- Code: `chessrows.py` next to this file. `chessrows.decision_input(board)` builds a request, where `board` is a
  `chess.Board` (python-chess 1.11.2). `chessrows.training_row(position, 0.02)` adds the training-only fields (id,
  family, target, label, source).
- Board representation: FEN only.
- When you call `jeff-serve`, the request object below goes inside its request body: the `state` as is, and the
  `question` under a name of your choice in `questions` (see the model card).

## The request object

```json
{"state": {"fen": "<FEN>"},
 "question": {"type": "choice", "instructions": "<instructions>", "criteria": {"<SAN move>": null, "...": null}}}
```

### `state`

One key only:

| key | value |
|---|---|
| `fen` | The full six-field Forsyth-Edwards Notation string of the position, exactly as python-chess `board.fen()` writes it: piece placement, side to move (`w`/`b`), castling rights (`KQkq` subset or `-`), en passant square, halfmove clock, fullmove number. |

Detail that matters: python-chess writes an en passant square **only when an en passant capture is actually legal** in the
position; otherwise the field is `-`. (Some libraries write the square after every two-square pawn push. If the web
app's library does that, replace the square with `-` unless a legal en passant capture exists.) The halfmove clock and
fullmove number are the real game counters.

### `instructions`

Exactly one of these two strings (depending on the side to move):

- White to move: `You are playing white. Choose the best move in this position. The position is given in Forsyth-Edwards Notation (FEN).`
- Black to move: `You are playing black. Choose the best move in this position. The position is given in Forsyth-Edwards Notation (FEN).`

### Options (`criteria`)

- One key per **legal move**, written in standard algebraic notation (SAN) exactly as python-chess `board.san(move)`
  writes it: piece letter `K Q R B N` (none for pawns), a disambiguating file or rank only when needed (`Nbd7`, `R1e2`),
  `x` for captures, `=Q`/`=R`/`=B`/`=N` for promotions, `O-O` / `O-O-O` for castling (capital letter O), and a
  suffix `+` for check or `#` for checkmate.
- Every value is `null` (no description). The model's prompt then shows each option as `<code>: <SAN>`, for example `A: Na3`.
- **Order:** sorted by plain character-code order of the SAN strings (Python `sorted()`, the same as JavaScript's
  default `Array.prototype.sort()` for these ASCII strings). Capital letters come before small letters, so piece moves
  and castling come first (`B…`, `K…`, `N…`, `O-O`, `Q…`, `R…`), then pawn moves (`a3`, `a4`, `b3`, …).
  During training the option order was shuffled at every step, so the model does not depend on the order, but use
  this order so requests match the evaluated setup exactly.
- The answer: the model returns one probability per option; play the option with the highest probability.
  Only legal moves are offered, so the model cannot play an illegal move. A Jeff question allows at most 255 options
  (a chess position has at most 218 legal moves), and `jeff-serve` refuses questions with more options than the model
  saw in training: 101 for the released model (`max_options` in its `decision_config.json`). A position with no legal
  moves (checkmate or stalemate) is an error: do not send it.

## Training-only fields

`target`: a probability per option, in the same order as `criteria`. For each move, Stockfish 19 (depth 10,
MultiPV = all legal moves, at most 4,000,000 nodes) gives a score from the mover's point of view; it becomes a win
chance `1 / (1 + exp(-0.00368208 * centipawns))` (a forced mate for the mover = 1, against = 0), and the targets are a
softmax of those win chances at temperature 0.02. `label`: the move with the highest target. A web app does not send
these.

Both worked examples below come from `worked_examples.py` (Stockfish 19, temperature 0.02); the tests in
`tests/test_chess_example.py` check that `chessrows.py` still builds these two requests exactly.

## Worked example 1 (start): rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

Request (what the web app sends):

```json
{
  "state": {
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  },
  "question": {
    "type": "choice",
    "instructions": "You are playing white. Choose the best move in this position. The position is given in Forsyth-Edwards Notation (FEN).",
    "criteria": {
      "Na3": null,
      "Nc3": null,
      "Nf3": null,
      "Nh3": null,
      "a3": null,
      "a4": null,
      "b3": null,
      "b4": null,
      "c3": null,
      "c4": null,
      "d3": null,
      "d4": null,
      "e3": null,
      "e4": null,
      "f3": null,
      "f4": null,
      "g3": null,
      "g4": null,
      "h3": null,
      "h4": null
    }
  }
}
```

Full training row (with the Stockfish target; `id`, `family` and `source` are example values):

```json
{
  "id": "example",
  "suite": "chess",
  "family": "example-example",
  "state": {
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  },
  "question": {
    "type": "choice",
    "instructions": "You are playing white. Choose the best move in this position. The position is given in Forsyth-Edwards Notation (FEN).",
    "criteria": {
      "Na3": null,
      "Nc3": null,
      "Nf3": null,
      "Nh3": null,
      "a3": null,
      "a4": null,
      "b3": null,
      "b4": null,
      "c3": null,
      "c4": null,
      "d3": null,
      "d4": null,
      "e3": null,
      "e4": null,
      "f3": null,
      "f4": null,
      "g3": null,
      "g4": null,
      "h3": null,
      "h4": null
    }
  },
  "target": [
    0.00238165504377509,
    0.05366178050474248,
    0.09321147066426218,
    0.004109911202111465,
    0.03888178704137457,
    0.01778989823059274,
    0.0204199413259946,
    0.009357099188732471,
    0.05883577649744263,
    0.07754492517744888,
    0.05124793833901568,
    0.0890210823939883,
    0.09321147066426218,
    0.25599644473648697,
    0.0019868449945870115,
    0.02238660570933898,
    0.07072724219408945,
    0.00022302629388265397,
    0.019502549898935847,
    0.019502549898935847
  ],
  "label": "e4",
  "source": {
    "dataset": "worked-example",
    "stockfish_depth": 10,
    "softmax_temperature": 0.02
  }
}
```

## Worked example 2 (middlegame): r1bq1rk1/pp1nbppp/2p1pn2/6B1/2BP4/2N1PN2/PP3PPP/2RQK2R b K - 0 9

Queen's Gambit Declined after 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O 6.Nf3 Nbd7 7.Rc1 c6 8.Bd3 dxc4 9.Bxc4, black to move.

Request:

```json
{
  "state": {
    "fen": "r1bq1rk1/pp1nbppp/2p1pn2/6B1/2BP4/2N1PN2/PP3PPP/2RQK2R b K - 0 9"
  },
  "question": {
    "type": "choice",
    "instructions": "You are playing black. Choose the best move in this position. The position is given in Forsyth-Edwards Notation (FEN).",
    "criteria": {
      "Ba3": null,
      "Bb4": null,
      "Bc5": null,
      "Bd6": null,
      "Kh8": null,
      "Nb6": null,
      "Nb8": null,
      "Nc5": null,
      "Nd5": null,
      "Ne4": null,
      "Ne5": null,
      "Ne8": null,
      "Ng4": null,
      "Nh5": null,
      "Qa5": null,
      "Qb6": null,
      "Qc7": null,
      "Qe8": null,
      "Rb8": null,
      "Re8": null,
      "a5": null,
      "a6": null,
      "b5": null,
      "b6": null,
      "c5": null,
      "e5": null,
      "g6": null,
      "h5": null,
      "h6": null
    }
  }
}
```

Full training row:

```json
{
  "id": "example",
  "suite": "chess",
  "family": "example-example",
  "state": {
    "fen": "r1bq1rk1/pp1nbppp/2p1pn2/6B1/2BP4/2N1PN2/PP3PPP/2RQK2R b K - 0 9"
  },
  "question": {
    "type": "choice",
    "instructions": "You are playing black. Choose the best move in this position. The position is given in Forsyth-Edwards Notation (FEN).",
    "criteria": {
      "Ba3": null,
      "Bb4": null,
      "Bc5": null,
      "Bd6": null,
      "Kh8": null,
      "Nb6": null,
      "Nb8": null,
      "Nc5": null,
      "Nd5": null,
      "Ne4": null,
      "Ne5": null,
      "Ne8": null,
      "Ng4": null,
      "Nh5": null,
      "Qa5": null,
      "Qb6": null,
      "Qc7": null,
      "Qe8": null,
      "Rb8": null,
      "Re8": null,
      "a5": null,
      "a6": null,
      "b5": null,
      "b6": null,
      "c5": null,
      "e5": null,
      "g6": null,
      "h5": null,
      "h6": null
    }
  },
  "target": [
    9.748287295138077e-11,
    0.004250362778883829,
    1.385724444814494e-10,
    0.002982401515696413,
    0.009497993558474074,
    0.015590442456299928,
    0.004250362778883829,
    1.3283977524074064e-10,
    0.12128925654896831,
    2.3469352458984733e-10,
    5.446799686096044e-10,
    0.015590442456299928,
    0.02045334097862724,
    0.012441462923019599,
    0.015590442456299928,
    0.018682064638710316,
    0.002982401515696413,
    0.010391265633408179,
    0.017855377225000624,
    0.04431012201804978,
    0.04233433420839518,
    0.15259078643473922,
    0.08798367882294723,
    0.08028037913730844,
    0.1329517076212582,
    3.908920005989693e-06,
    0.0037207075878845203,
    0.0005966244050053752,
    0.1833801322318688
  ],
  "label": "h6",
  "source": {
    "dataset": "worked-example",
    "stockfish_depth": 10,
    "softmax_temperature": 0.02
  }
}
```

A real training row looks the same, with `"id": "chess-train-puzzle-124682"`, `"family": "puzzle-QYhT8"` (or
`"game-<lichess game id>"`) and a `source` such as
`{"dataset": "lichess-puzzles", "upstream_id": "QYhT8", "rating": 2168, "themes": "...", "license": "CC0", "stockfish_depth": 10, "softmax_temperature": 0.02}`.

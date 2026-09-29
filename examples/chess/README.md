# Chess fine-tune example

This folder holds the code that made [Jeff-Qwen3.5-0.8B-Chess](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B-Chess):
Jeff-Qwen3.5-0.8B fine-tuned to pick chess moves. The model reads a position in FEN and gives every legal move a
probability in one forward pass, with no search. It shows the usual way to use Jeff when zero-shot is not enough:
build rows in Jeff's request format from your own data, then run a short full-weight fine-tune with `jeff-train`.

Training had two stages, one epoch each, both starting a fresh optimizer:

1. **Stage one:** 300,000 positions, 60% Lichess puzzles and 40% positions from Lichess games of January 2013,
   starting from Jeff-Qwen3.5-0.8B.
2. **Stage two:** 300,000 new positions, 30% puzzles and 70% positions from August 2026 games in which both players are
   rated 1800 or more, starting from the stage-one checkpoint. **The released model is the stage-two checkpoint.**

Stockfish 19 scored every legal move at depth 10 (at most 4,000,000 nodes per position). Each score becomes a win
probability with Lichess's formula, and a softmax at temperature 0.02 turns those into the training target. The request
format is documented in [PROMPT_FORMAT.md](PROMPT_FORMAT.md).

## Results (released model)

| Measure | Result |
|---|---|
| 1,000 held-out Lichess puzzles, top move = first solution move | **55.8%** (95% interval 52.7–58.9%) |
| By puzzle rating: under 1000 / 1000–1499 / 1500–1999 / 2000+ | 86.4% / 62.0% / 40.4% / 34.4% |
| Same puzzles: random legal move / untrained Qwen3.5-0.8B / Jeff-Qwen3.5-0.8B zero-shot / mate-capture-check rule | 5.1% / 6.2% / 15.5% / 37.9% |
| 20 games each against Stockfish at UCI_Elo 1350 / 1600 / 1900 (wins–draws–losses) | 0–0–20 / 0–3–17 / 0–1–19 |
| Estimated rating, Stockfish UCI_Elo scale | about 1,000 |
| Time per move, one position at a time, RTX PRO 6000 | 23 ms |
| Batched throughput with 95% of answers under 200 ms (batch 32) | 214 moves per second, about 600 blitz games per GPU |
| 100 simultaneous games against human-paced Stockfish 1350 | 1 win, 30 draws, 69 losses; median response 40 ms, 95th percentile 74 ms |

The throughput benchmark, the 100-game run and its video used the stage-one checkpoint, which scored 56.0% on the
puzzles (stage two: 55.8%) and has the same speed. Stage two was better on ordinary game positions, so it was released.

## What each file does

| File | Step |
|---|---|
| `chessrows.py` | Turns a position into a Jeff request, a training row or a held-out puzzle row. The only place requests are built |
| `select_stage1.py` | Stage one: picks the test, training, development and calibration positions from the puzzles and one month of games |
| `select_stage2.py` | Stage two: picks 300,000 new training positions, disjoint from stage one, from games streamed on standard input |
| `label.py` | Scores every legal move of each position with Stockfish |
| `target_stats.py` | Shows how the softmax temperature spreads the target (how T = 0.02 was chosen) |
| `build_rows.py` | Writes `train.jsonl`, `dev.jsonl`, `calibration.jsonl` and `test.jsonl` for a stage |
| `train.sh` | One training stage with `jeff-train` and the release settings |
| `score.py` | Puzzle accuracy by rating band, the random and rule floors, and top-3 accuracy |
| `play.py` | Games against Stockfish at a fixed UCI_Elo (or a random mover) |
| `elo.py` | Rating estimate from those games |
| `speed.py` | Milliseconds per move, one position at a time |
| `human_times.py` | Human time per move from the clocks in Lichess blitz games |
| `move_server.py` | A batching move server, and the throughput benchmark |
| `simul.py` | Many games at once against Stockfish that waits a human think time before each move |
| `render.py` | The full video of the simultaneous run, and a GIF preview |
| `featured.py`, `render_featured.py` | Jeff's move probabilities in one game of the run, and the short video on the model card |
| `worked_examples.py` | Prints the two worked examples in `PROMPT_FORMAT.md` |

## Reproduce

You need a Linux machine with an NVIDIA GPU for training, evaluation, games and speed (the release used one RTX PRO
6000 and a 16-core, 32-thread CPU running 28 Stockfish processes), and these tools: `uv`, `curl`, `zstd`, `ffmpeg` (videos
only) and Stockfish 19. Run every command from the repository root. The data, runs and checkpoints go to `data/chess`,
`runs/chess` and `checkpoints/chess`, which git ignores.

**1. Install.**

```bash
uv sync --extra chess --extra cuda      # python-chess, zstandard and the fast Qwen3.5 kernels
mkdir -p data/chess/raw bin
curl -L https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz \
  | tar -xz -C bin
SF=bin/stockfish/stockfish-linux-x86-64-universal
$SF bench 2>&1 | tail -1      # check that it runs
uv run hf download mstrasser/Jeff-Qwen3.5-0.8B --local-dir checkpoints/jeff-0.8b
```

**2. Stage-one data.** The Lichess puzzle database and the January 2013 games (both CC0). Selecting takes a few
minutes; labelling the 304,000 training, development and calibration positions took 46 minutes.

```bash
curl -L https://database.lichess.org/lichess_db_puzzle.csv.zst -o data/chess/raw/lichess_db_puzzle.csv.zst
curl -L https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst \
  -o data/chess/raw/lichess_db_standard_rated_2013-01.pgn.zst
uv run python examples/chess/select_stage1.py --puzzles data/chess/raw/lichess_db_puzzle.csv.zst \
  --games data/chess/raw/lichess_db_standard_rated_2013-01.pgn.zst --output data/chess/positions-stage1.jsonl
uv run python examples/chess/label.py --positions data/chess/positions-stage1.jsonl \
  --output data/chess/labels-stage1.jsonl --engine $SF --depth 10 --node-cap 4000000
uv run python examples/chess/target_stats.py --labels data/chess/labels-stage1.jsonl \
  --temperatures 0.005 0.01 0.02 0.03 0.05 0.1      # optional: how the temperature was chosen
uv run python examples/chess/build_rows.py stage1 --positions data/chess/positions-stage1.jsonl \
  --labels data/chess/labels-stage1.jsonl --temperature 0.02 --output data/chess/stage1
```

**3. Stage-one training** (1 hour 41 minutes on the RTX PRO 6000). It starts from the published Jeff-Qwen3.5-0.8B,
the same weights the release started from. The checkpoint with the lowest development loss is linked as
`checkpoints/chess/chess-0.8b-stage1/selected`.

```bash
examples/chess/train.sh chess-0.8b-stage1 data/chess/stage1 checkpoints/jeff-0.8b
```

**4. Stage-two data.** The August 2026 games are streamed; the script stops reading after 190,000 games in which both
players are rated 1800 or more, so `curl` and `zstd` then end with a broken pipe, which is expected. Only the Python
step must succeed. `build_rows.py stage2` copies the stage-one development, calibration and test files unchanged.

```bash
curl -sL https://database.lichess.org/standard/lichess_db_standard_rated_2026-08.pgn.zst | zstd -dc \
  | uv run python examples/chess/select_stage2.py --puzzles data/chess/raw/lichess_db_puzzle.csv.zst \
      --stage-one data/chess/positions-stage1.jsonl --output data/chess/positions-stage2.jsonl
uv run python examples/chess/label.py --positions data/chess/positions-stage2.jsonl \
  --output data/chess/labels-stage2.jsonl --engine $SF --depth 10 --node-cap 4000000
uv run python examples/chess/build_rows.py stage2 --labels data/chess/labels-stage2.jsonl \
  --stage-one data/chess/stage1 --temperature 0.02 --output data/chess/stage2
```

**5. Stage-two training** (1 hour 40 minutes). Its `selected` checkpoint is the released model.

```bash
examples/chess/train.sh chess-0.8b-stage2 data/chess/stage2 checkpoints/chess/chess-0.8b-stage1/selected
```

**6. Held-out puzzles.** Score the two stages and the two baselines on the same 1,000 puzzles, then break the results
down by rating band. `--random` and `--rule` add the two floors that need no model.

```bash
TEST=data/chess/stage1/test.jsonl
uv run jeff-evaluate --local --checkpoint checkpoints/chess/chess-0.8b-stage2/selected --data $TEST --output runs/chess/eval/stage2.json
uv run jeff-evaluate --local --checkpoint checkpoints/chess/chess-0.8b-stage1/selected --data $TEST --output runs/chess/eval/stage1.json
uv run jeff-evaluate --local --checkpoint checkpoints/jeff-0.8b --data $TEST --output runs/chess/eval/jeff-0.8b.json
uv run jeff-evaluate --local --base-model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --data $TEST --output runs/chess/eval/qwen-0.8b-untrained.json
uv run python examples/chess/score.py --test $TEST --random --rule --predictions runs/chess/eval/*.predictions.jsonl
```

**7. Games against Stockfish and the rating.** 20 games at each strength, 10 with each colour, 0.2 s per Stockfish
move. The rating is a maximum-likelihood Elo fit with one virtual draw per opponent level, so it stays finite when the
model loses almost every game; treat it as "clearly below Stockfish's 1350 setting".

```bash
CK=checkpoints/chess/chess-0.8b-stage2/selected
mkdir -p runs/chess/games
for elo in 1350 1600 1900; do
  uv run python examples/chess/play.py --checkpoint $CK --engine $SF --elo $elo --games 20 --output runs/chess/games/stage2-elo$elo.json
done
uv run python examples/chess/play.py --checkpoint checkpoints/jeff-0.8b --engine $SF --elo 1350 --games 20 \
  --output runs/chess/games/jeff-0.8b-elo1350.json
uv run python examples/chess/elo.py runs/chess/games/stage2-elo1350.json runs/chess/games/stage2-elo1600.json runs/chess/games/stage2-elo1900.json
```

**8. Speed, human move times and throughput.** Time the model alone on the GPU, with nothing else running.
`human_times.py` reads the clocks of the first 20,000 rated blitz games (3+0, 3+2, 5+0, 5+3) in the August 2026 file.
The release ran the throughput benchmark and the simultaneous run (step 9) with the stage-one checkpoint, `CK1`.

```bash
CK1=checkpoints/chess/chess-0.8b-stage1/selected
mkdir -p runs/chess/blitz
uv run python examples/chess/speed.py --checkpoint $CK --positions data/chess/positions-stage1.jsonl
U=https://database.lichess.org/standard/lichess_db_standard_rated_2026-08.pgn.zst
curl -sL $U | zstd -dc | uv run python examples/chess/human_times.py --games 20000 --source $U \
  --output runs/chess/blitz/human-times.jsonl --summary runs/chess/blitz/human-times-summary.json
uv run python examples/chess/move_server.py benchmark --checkpoint $CK1 --positions data/chess/positions-stage2.jsonl \
  --output runs/chess/blitz/throughput.json
```

Games per GPU ≈ moves per second at the largest batch whose 95th-percentile latency stays under 200 ms, times the
median human time per move (214 × 3.0 s ≈ 640 in the release, stated as about 600).

**9. The simultaneous run and its videos.** 100 games at once; each opponent is Stockfish at UCI_Elo 1350 (one thread,
0.2 s per search, a pool of 24 processes) that releases its move only after a think time drawn from the human times.
The renderers run on the CPU and need `ffmpeg` and a font with the Unicode chess symbols (DejaVu Sans on Linux; on a Mac,
Arial Unicode for the pieces). The model-card video features game 63 (number 62 in the log).

```bash
uv run python examples/chess/simul.py --checkpoint $CK1 --engine $SF --human-times runs/chess/blitz/human-times.jsonl \
  --games 100 --engines 24 --elo 1350 --move-time 0.2 --max-batch 128 --cap 900 \
  --log runs/chess/blitz/simul-log.jsonl --summary runs/chess/blitz/simul-summary.json
FONT=/usr/share/fonts/truetype/dejavu
uv run python examples/chess/render.py --log runs/chess/blitz/simul-log.jsonl --output runs/chess/blitz/jeff-100-blitz.mp4 \
  --gif runs/chess/blitz/jeff-100-blitz-preview.gif --gif-start 20 --fps 10 --speed 1 \
  --font $FONT/DejaVuSans.ttf --bold-font $FONT/DejaVuSans-Bold.ttf \
  --caption "Jeff chess (0.8B, no search) vs 100 opponents at once: Stockfish UCI_Elo 1350 moving after human think times sampled from real Lichess blitz games. One RTX PRO 6000."
uv run python examples/chess/featured.py --log runs/chess/blitz/simul-log.jsonl --game 62 --checkpoint $CK1 \
  --output runs/chess/blitz/featured-62.json
uv run python examples/chess/render_featured.py --log runs/chess/blitz/simul-log.jsonl \
  --featured runs/chess/blitz/featured-62.json --game 62 --games-per-gpu 600 \
  --label "Game 63: Jeff's one win out of 100 (checkmate in 9 moves)" --output runs/chess/blitz/jeff-100-blitz-game63.mp4 \
  --font $FONT/DejaVuSans.ttf --bold-font $FONT/DejaVuSans-Bold.ttf --piece-font $FONT/DejaVuSans.ttf
```

The game results of a new run will differ: Stockfish's weakened play is random, and the timing of the opponents
changes which boards share a batch. Choose the game to feature, and its label, from your own log.

## What to expect when you rerun it

- The selection scripts are seeded and give the same positions from the same files. Stockfish at a fixed depth with a
  cleared hash gives the same scores, so the training rows are the same. `label.py` writes rows in the order they
  finish, so the order of `train.jsonl` (and so the training batches) can differ, and the accuracy can move by a
  point or so.
- The August 2026 selection takes the first qualifying games in the file, so it depends on Lichess keeping that file
  unchanged.
- The rating estimate rests on a handful of draws: expect it to move by 100 points or more between runs.

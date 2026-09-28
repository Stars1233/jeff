#!/usr/bin/env bash
# The five game tests from the README (20 episodes each, seed 1234, a video of episode 1) against a running Jeff server.
# Usage: scripts/games_all.sh URL NAME     e.g. scripts/games_all.sh http://127.0.0.1:8765 jeff-0.8b
# Needs the games extra: uv pip install -e ".[games]"
set -euo pipefail
cd "$(dirname "$0")/.."
url=${1:?usage: scripts/games_all.sh URL NAME}; name=${2:?usage: scripts/games_all.sh URL NAME}
for test in "doom situation" "doom outcomes" "doom rule" "frogger outcomes" "pacman outcomes"; do
  set -- $test
  uv run python -m jeff.games --game "$1" --player jeff --criteria "$2" --url "$url" --video --out "runs/games/$1-$name-$2.json"
done
for game in doom frogger pacman; do
  for player in random rule; do
    test -e "runs/games/$game-$player.json" || uv run python -m jeff.games --game $game --player $player --out "runs/games/$game-$player.json"
  done
done

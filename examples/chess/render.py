"""Render the simultaneous-games log (simul.py) as an MP4 video, plus a short GIF preview.

All boards are shown in a grid; each board updates when its moves happen in the log. Jeff's moves are highlighted
(from- and to-square) for --highlight seconds. A header strip shows the elapsed time, games in progress, Jeff's
running win/draw/loss score, moves Jeff has answered, and Jeff's median and 95th-percentile response time so far.
A caption line states what the opponents are. Frames are drawn with Pillow and piped to ffmpeg (H.264, yuv420p,
+faststart). --speed 1 is real time; any other value is written into the header as "N x speed".
Needs ffmpeg on PATH and a font with the Unicode chess symbols (for example DejaVu Sans). CPU only.
"""

import argparse
import bisect
import json
import math
import subprocess
from pathlib import Path

import chess
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
HEADER, FOOTER = 92, 40
LIGHT, DARK = (238, 238, 210), (118, 150, 86)
HIGHLIGHT_LIGHT, HIGHLIGHT_DARK = (246, 246, 105), (186, 202, 43)
BACKGROUND = (24, 26, 30)
TEXT = (235, 235, 235)
MUTED = (160, 165, 175)
RESULT_COLOURS = {"win": (80, 200, 120), "draw": (170, 170, 170), "loss": (220, 90, 90)}
GLYPHS = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}


def layout(count: int) -> tuple[int, int, int]:
    """Columns, rows and board size (pixels, including a small label strip) that fit `count` boards largest."""
    area_height = HEIGHT - HEADER - FOOTER
    best = (0, 0, 0)
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        size = min(WIDTH // columns, area_height // rows)
        if size > best[2]:
            best = (columns, rows, size)
    return best


class Renderer:
    def __init__(self, games: int, jeff_colours: dict[int, str], font: Path, bold: Path) -> None:
        self.columns, self.rows, self.cell = layout(games)
        self.board_px = (self.cell - 16) // 8 * 8
        self.square = self.board_px // 8
        self.left = (WIDTH - self.columns * self.cell) // 2
        self.top = HEADER + (HEIGHT - HEADER - FOOTER - self.rows * self.cell) // 2
        self.piece_font = ImageFont.truetype(font, int(self.square * 1.1))
        self.small = ImageFont.truetype(font, 11)
        self.header_font = ImageFont.truetype(bold, 30)
        self.header_small = ImageFont.truetype(font, 20)
        self.caption_font = ImageFont.truetype(font, 20)
        self.image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)
        self.jeff_colours = jeff_colours

    def origin(self, game: int) -> tuple[int, int]:
        column, row = game % self.columns, game // self.columns
        x = self.left + column * self.cell + (self.cell - self.board_px) // 2
        y = self.top + row * self.cell + 2
        return x, y

    def board(self, game: int, board: chess.Board, highlight: chess.Move | None, result: str | None, moves: int) -> None:
        x0, y0 = self.origin(game)
        draw = self.draw
        draw.rectangle([x0 - 2, y0 - 2, x0 + self.board_px + 1, y0 + self.board_px + 13], fill=BACKGROUND)
        jeff_white = self.jeff_colours[game] == "white"
        lit = {highlight.from_square, highlight.to_square} if highlight else set()
        for square in chess.SQUARES:
            file, rank = chess.square_file(square), chess.square_rank(square)
            # Each board is drawn from Jeff's side.
            column, row = (file, 7 - rank) if jeff_white else (7 - file, rank)
            light = (file + rank) % 2 == 1
            colour = (HIGHLIGHT_LIGHT if light else HIGHLIGHT_DARK) if square in lit else (LIGHT if light else DARK)
            sx, sy = x0 + column * self.square, y0 + row * self.square
            draw.rectangle([sx, sy, sx + self.square - 1, sy + self.square - 1], fill=colour)
            piece = board.piece_at(square)
            if piece:
                white = piece.color == chess.WHITE
                draw.text((sx + self.square / 2, sy + self.square / 2 + 1), GLYPHS[piece.symbol().lower()],
                          font=self.piece_font, anchor="mm", fill=(255, 255, 255) if white else (10, 10, 10),
                          stroke_width=1, stroke_fill=(0, 0, 0) if white else (90, 90, 90))
        if result:
            draw.rectangle([x0 - 2, y0 - 2, x0 + self.board_px + 1, y0 + self.board_px + 1], outline=RESULT_COLOURS[result], width=3)
        label = f"#{game + 1} Jeff {'W' if jeff_white else 'B'} · {moves}" + (f" · {result}" if result else "")
        draw.text((x0, y0 + self.board_px + 2), label, font=self.small, fill=RESULT_COLOURS[result] if result else MUTED)

    def header(self, elapsed: float, speed: float, running: int, total: int, score: dict, answered: int,
               median_ms: float | None, p95_ms: float | None) -> None:
        draw = self.draw
        draw.rectangle([0, 0, WIDTH, HEADER - 1], fill=(36, 39, 46))
        minutes, seconds = divmod(int(elapsed), 60)
        pace = "real time" if speed == 1 else f"{speed:g}x speed"
        latency = "–" if median_ms is None else f"{median_ms:.0f} ms median · {p95_ms:.0f} ms p95"
        items = [("Elapsed", f"{minutes:02d}:{seconds:02d} ({pace})"), ("Games in progress", f"{running} / {total}"),
                 ("Jeff W / D / L", f"{score['win']} / {score['draw']} / {score['loss']}"),
                 ("Moves Jeff answered", f"{answered}"), ("Jeff response time", latency)]
        widths = [330, 300, 280, 330, 480]
        x = 30
        for (title, value), width in zip(items, widths):
            draw.text((x, 12), title, font=self.header_small, fill=MUTED)
            draw.text((x, 40), value, font=self.header_font, fill=TEXT)
            x += width

    def caption(self, text: str) -> None:
        self.draw.rectangle([0, HEIGHT - FOOTER, WIDTH, HEIGHT], fill=BACKGROUND)
        self.draw.text((WIDTH // 2, HEIGHT - FOOTER // 2), text, font=self.caption_font, anchor="mm", fill=MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="simul.py --log")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gif", type=Path, required=True)
    parser.add_argument("--gif-start", type=float, default=60, help="Video second where the 12-second GIF starts")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--highlight", type=float, default=1.0, help="Seconds a Jeff move stays highlighted (log time)")
    parser.add_argument("--caption", required=True)
    parser.add_argument("--font", type=Path, required=True, help="Regular TrueType font with the chess symbols, e.g. DejaVuSans.ttf")
    parser.add_argument("--bold-font", type=Path, required=True, help="Bold TrueType font, e.g. DejaVuSans-Bold.ttf")
    args = parser.parse_args()
    events = [json.loads(line) for line in args.log.open()]
    config = events[0]
    if config["kind"] != "config":
        raise ValueError("The log must start with its config line")
    games = config["games"]
    jeff_colours = {e["game"]: e["jeff"] for e in events if e["kind"] == "start"}
    stop = next(e["t"] for e in events if e["kind"] == "stop")
    renderer = Renderer(games, jeff_colours, args.font, args.bold_font)
    boards = {game: chess.Board() for game in range(games)}
    results: dict[int, str] = {}
    last_jeff: dict[int, tuple[chess.Move, float]] = {}
    moves_done = {game: 0 for game in range(games)}
    latencies: list[float] = []
    score = {"win": 0, "draw": 0, "loss": 0}
    for game in range(games):
        renderer.board(game, boards[game], None, None, 0)
    renderer.caption(args.caption)
    frames = math.ceil(stop / args.speed * args.fps) + args.fps * 3  # hold the last frame for three seconds
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
                               "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                               str(args.output)], stdin=subprocess.PIPE)
    index = 1
    for frame in range(frames):
        now = min(frame / args.fps * args.speed, stop)
        changed = set()
        while index < len(events) and events[index]["t"] <= now:
            event = events[index]
            index += 1
            if event["kind"] == "move":
                game = event["game"]
                move = chess.Move.from_uci(event["uci"])
                if move not in boards[game].legal_moves:
                    raise ValueError(f"Illegal move in the log: game {game} {event['uci']}")
                boards[game].push(move)
                moves_done[game] += 1
                if event["by"] == "jeff":
                    last_jeff[game] = (move, event["t"])
                    bisect.insort(latencies, event["latency_ms"])
                changed.add(game)
            elif event["kind"] == "end":
                results[event["game"]] = event["result"]
                score[event["result"]] += 1
                changed.add(event["game"])
        for game, (move, when) in list(last_jeff.items()):
            if now - when > args.highlight:
                del last_jeff[game]
                changed.add(game)
        for game in changed:
            highlight = last_jeff.get(game, (None, 0))[0]
            renderer.board(game, boards[game], highlight, results.get(game), moves_done[game])
        median = latencies[len(latencies) // 2] if latencies else None
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None
        renderer.header(now, args.speed, games - len(results), games, score, len(latencies), median, p95)
        ffmpeg.stdin.write(renderer.image.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing the video")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(args.gif_start), "-t", "12", "-i", str(args.output),
                    "-vf", "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                    str(args.gif)], check=True)
    print(json.dumps({"video": str(args.output), "gif": str(args.gif), "frames": frames, "fps": args.fps,
                      "speed": args.speed, "final_score": score, "unfinished": games - len(results)}))


if __name__ == "__main__":
    main()

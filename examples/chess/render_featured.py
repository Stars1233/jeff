"""A short video of the simultaneous run: a title card, one featured game in real time (with Jeff's top three moves and
their probabilities) beside the other 99 boards, and an end card. This is the video on the model card.

Inputs: the simul.py log (every move with its time) and the featured.py output for the featured game. The video shows
the whole featured game, then holds its final position for 1.5 seconds. Frames are drawn with Pillow and piped to
ffmpeg (H.264, yuv420p, +faststart), so ffmpeg must be on PATH. CPU only.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import chess
from PIL import Image, ImageDraw, ImageFont

FPS = 25
TITLE_S, END_S = 1.5, 2.5
JEFF_BANNER_S = 0.8  # how long "Jeff: N ms" shows after Jeff moves
W, H = 1920, 1080

BG, PANEL, TEXT, MUTED = (17, 19, 24), (27, 30, 37), (236, 238, 242), (150, 154, 166)
LIGHT, DARK = (235, 236, 208), (119, 149, 86)
MOVE_TINT = (246, 246, 105)
BLUE, GREY = (79, 134, 224), (78, 82, 94)
GLYPH = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜", chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}


class Fonts:
    def __init__(self, sans: Path, bold: Path, pieces: Path) -> None:
        self.paths = {"sans": sans, "bold": bold, "pieces": pieces}
        self.cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def __call__(self, kind: str, size: int) -> ImageFont.FreeTypeFont:
        if (kind, size) not in self.cache:
            self.cache[kind, size] = ImageFont.truetype(self.paths[kind], size)
        return self.cache[kind, size]


def draw_board(font: Fonts, size: int, board: chess.Board, last: chess.Move | None, flip: bool) -> Image.Image:
    square = size / 8
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    pieces = font("pieces", int(square * 0.86))
    stroke = max(1, int(square / 22))
    for sq in chess.SQUARES:
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        col, row = (7 - file, rank) if flip else (file, 7 - rank)
        x0, y0 = round(col * square), round(row * square)
        x1, y1 = round((col + 1) * square), round((row + 1) * square)
        colour = LIGHT if (file + rank) % 2 else DARK
        if last is not None and sq in (last.from_square, last.to_square):
            colour = tuple((a + b) // 2 for a, b in zip(colour, MOVE_TINT))
        draw.rectangle([x0, y0, x1, y1], fill=colour)
        piece = board.piece_at(sq)
        if piece is not None:
            white = piece.color == chess.WHITE
            draw.text(((x0 + x1) / 2, (y0 + y1) / 2 + square * 0.03), GLYPH[piece.piece_type], font=pieces, anchor="mm",
                      fill=(255, 255, 255) if white else (20, 20, 20), stroke_width=stroke,
                      stroke_fill=(30, 30, 30) if white else (230, 230, 230))
    return image


def featured_banner(board: chess.Board, jeff: str, now: float, jeff_moved_at: float | None,
                    decision: dict | None) -> tuple[str, tuple] | None:
    """What to show over the featured board: the result, Jeff's reply time just after it moves, or a blinking
    "Opponent thinking..." while it is the opponent's turn."""
    if board.is_checkmate():
        jeff_won = (board.turn == chess.BLACK) == (jeff == "white")
        return ("Checkmate: Jeff wins" if jeff_won else "Checkmate: Jeff loses", BLUE if jeff_won else GREY)
    if jeff_moved_at is not None and decision is not None and now - jeff_moved_at < JEFF_BANNER_S:
        return (f"Jeff: {decision['latency_ms']:.0f} ms", BLUE)
    jeff_to_move = (board.turn == chess.WHITE) == (jeff == "white")
    if not jeff_to_move and int(now * 2) % 2 == 0:
        return ("Opponent thinking...", GREY)
    return None


def banner(font: Fonts, image: Image.Image, x: int, y: int, size: int, content: tuple[str, tuple] | None) -> None:
    if content is None:
        return
    text, colour = content
    band = Image.new("RGBA", (size, 96), colour + (215,))
    ImageDraw.Draw(band).text((size / 2, 48), text, font=font("bold", 44), fill=(255, 255, 255), anchor="mm")
    image.paste(band, (x, y + size // 2 - 48), band)


def card(font: Fonts, lines: list[tuple[str, int, tuple, bool]], alpha: float) -> Image.Image:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    y = H / 2 - sum(size * 1.6 for _, size, _, _ in lines) / 2
    for text, size, colour, bold in lines:
        y += size * 0.8
        faded = tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(colour, BG))
        draw.text((W / 2, y), text, font=font("bold" if bold else "sans", size), fill=faded, anchor="mm")
        y += size * 0.8
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True, help="simul.py --log")
    parser.add_argument("--featured", type=Path, required=True, help="featured.py output for the featured game")
    parser.add_argument("--game", type=int, required=True, help="The featured game's number in the log (0-based)")
    parser.add_argument("--label", required=True, help="Caption under the featured board")
    parser.add_argument("--games-per-gpu", type=int, required=True, help="Capacity estimate for the end card (the release said 600)")
    parser.add_argument("--output", type=Path, required=True, help="MP4 file to write")
    parser.add_argument("--font", type=Path, required=True, help="Regular TrueType font, e.g. Arial.ttf")
    parser.add_argument("--bold-font", type=Path, required=True, help="Bold TrueType font, e.g. Arial Bold.ttf")
    parser.add_argument("--piece-font", type=Path, required=True, help="Font with the Unicode chess symbols, e.g. Arial Unicode.ttf")
    args = parser.parse_args()
    font = Fonts(args.font, args.bold_font, args.piece_font)

    colours: dict[int, str] = {}
    moves: list[dict] = []
    stop = None
    for row in map(json.loads, args.log.open()):
        if row["kind"] == "start":
            colours[row["game"]] = row["jeff"]
        elif row["kind"] == "move":
            moves.append(row)
        elif row["kind"] == "stop":
            stop = row["t"]
    if stop is None:
        raise ValueError(f"{args.log} has no stop line: the run did not finish")
    moves.sort(key=lambda m: m["t"])
    featured = json.loads(args.featured.read_text())
    if not featured:
        raise ValueError(f"No Jeff moves recorded in {args.featured}")
    if args.game not in colours:
        raise ValueError(f"Game {args.game} is not in {args.log}")
    others = [g for g in sorted(colours) if g != args.game]
    if len(others) != 99:
        raise ValueError(f"The layout holds 99 other games; the log has {len(others)}")
    run_latencies = [m["latency_ms"] for m in moves if m["by"] == "jeff"]

    boards = {g: chess.Board() for g in colours}
    last_move: dict[int, chess.Move | None] = {g: None for g in colours}
    jeff_flash: dict[int, float] = {}
    small_cache: dict[int, Image.Image] = {}
    decision = None
    latencies: list[float] = []
    answered = 0

    small, cell, grid_x, grid_y, cols = 90, 98, 766, 150, 11
    big, big_x, big_y = 600, 70, 150
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)], stdin=subprocess.PIPE)
    frames = 0

    def emit(image: Image.Image) -> None:
        nonlocal frames
        ffmpeg.stdin.write(image.tobytes())
        frames += 1

    title = [("100 blitz games at once.", 64, TEXT, True), ("One GPU. One 0.8B model.", 64, TEXT, True),
             ("No search: one look at the board per move.", 34, MUTED, False)]
    for i in range(int(TITLE_S * FPS)):
        emit(card(font, title, min(1.0, (i + 1) / (0.5 * FPS))))

    live_s = featured[-1]["t"] + 1.5  # show the whole featured game, then a moment on the final position
    jeff_moved_at: float | None = None
    cursor, feat_index = 0, 0
    for i in range(int(live_s * FPS)):
        now = i / FPS
        while cursor < len(moves) and moves[cursor]["t"] <= now:
            m = moves[cursor]
            g = m["game"]
            move = chess.Move.from_uci(m["uci"])
            boards[g].push(move)
            last_move[g] = move
            small_cache.pop(g, None)
            if m["by"] == "jeff":
                jeff_flash[g] = now
                latencies.append(m["latency_ms"])
                answered += 1
            cursor += 1
        while feat_index < len(featured) and featured[feat_index]["t"] <= now:
            decision = featured[feat_index]
            jeff_moved_at = featured[feat_index]["t"]
            feat_index += 1

        image = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(image)
        # Header.
        draw.text((70, 38), "Jeff Chess", font=font("bold", 40), fill=TEXT, anchor="lm")
        draw.text((70, 84), "one 0.8B model · 100 blitz games at once · one GPU · real time", font=font("sans", 24),
                  fill=MUTED, anchor="lm")
        stats = [("elapsed", f"0:{int(now):02d}"), ("moves answered", f"{answered}"),
                 ("median reply", f"{statistics.median(latencies):.0f} ms" if latencies else "–")]
        x = W - 70
        for label, value in reversed(stats):
            draw.text((x, 40), value, font=font("bold", 40), fill=TEXT, anchor="rm")
            draw.text((x, 86), label, font=font("sans", 22), fill=MUTED, anchor="rm")
            x -= 290
        # Featured board and Jeff's decision.
        flip = colours[args.game] == "black"
        image.paste(draw_board(font, big, boards[args.game], last_move[args.game], flip), (big_x, big_y))
        banner(font, image, big_x, big_y, big,
               featured_banner(boards[args.game], colours[args.game], now, jeff_moved_at, decision))
        draw = ImageDraw.Draw(image)
        draw.text((big_x, big_y + big + 28), args.label, font=font("bold", 22), fill=TEXT, anchor="lm")
        if decision is not None:
            draw.text((big_x, big_y + big + 66),
                      f"Jeff's last move: one look at {decision['legal']} legal moves, {decision['latency_ms']:.0f} ms",
                      font=font("sans", 22), fill=MUTED, anchor="lm")
            for k, (san, p) in enumerate(decision["top"][:3]):
                y = big_y + big + 104 + k * 38
                played = san == decision["played"]
                draw.text((big_x, y), san, font=font("bold" if played else "sans", 24), fill=TEXT if played else MUTED,
                          anchor="lm")
                bar = int(430 * p)
                draw.rounded_rectangle([big_x + 110, y - 12, big_x + 110 + max(bar, 4), y + 12], radius=4,
                                       fill=BLUE if played else GREY)
                draw.text((big_x + 110 + max(bar, 4) + 12, y), f"{p:.0%}", font=font("sans", 22), fill=MUTED, anchor="lm")
        # The other 99 games.
        for index, g in enumerate(others):
            col, row = index % cols, index // cols
            x0, y0 = grid_x + col * cell, grid_y + row * cell
            if g not in small_cache:
                small_cache[g] = draw_board(font, small, boards[g], last_move[g], colours[g] == "black")
            image.paste(small_cache[g], (x0, y0))
            if g in jeff_flash and now - jeff_flash[g] < 0.5:
                draw.rectangle([x0 - 3, y0 - 3, x0 + small + 2, y0 + small + 2], outline=BLUE, width=3)
        draw.text((W / 2, H - 28), "Opponents: Stockfish at its weakest setting (UCI_Elo 1350), each move delayed by a "
                  "human think time sampled from real Lichess blitz games. Blue flash: Jeff has just moved.",
                  font=font("sans", 20), fill=MUTED, anchor="mm")
        emit(image)

    end = [(f"About {args.games_per_gpu} blitz games per GPU", 60, TEXT, True),
           (f"median reply {statistics.median(run_latencies):.0f} ms over the full {round(stop / 60)}-minute run", 34, MUTED, False),
           ("Jeff-Qwen3.5-0.8B-Chess", 40, BLUE, True)]
    for i in range(int(END_S * FPS)):
        emit(card(font, end, min(1.0, (i + 1) / (0.5 * FPS))))

    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing the video")
    print(json.dumps({"video": str(args.output), "frames": frames, "fps": FPS}))


if __name__ == "__main__":
    main()

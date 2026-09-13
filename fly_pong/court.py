"""Medieval hall for the Python court. Physics size unchanged. Not a fly title."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SCALE = 2
SIDE = 120
ASSETS = Path(__file__).resolve().parent / "assets"

OAK = (18, 12, 8)
STONE = (72, 54, 40)
STONE_DK = (48, 36, 26)
STONE_LT = (96, 74, 54)
BEAM = (36, 24, 16)
GOLD = (198, 156, 56)
GOLD_DK = (118, 86, 28)
CRIMSON = (138, 26, 26)
CRIMSON_DK = (78, 14, 14)
CREAM = (230, 210, 168)
IRON = (42, 36, 30)
WOOD = (118, 72, 34)
WOOD_DK = (68, 42, 18)
PARCHMENT = (228, 208, 158)
HIGHLIGHT = (240, 228, 196)


def frame_size(C: dict[str, Any]) -> tuple[int, int]:
    return int(C["width"]) * SCALE + 2 * SIDE, int(C["height"]) * SCALE


def court_from_window_y(mouse_y: float) -> float:
    return float(mouse_y) / float(SCALE)


def _fonts(pygame):
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
    return (
        pygame.font.Font(None, 42),
        pygame.font.Font(None, 32),
        pygame.font.Font(None, 22),
    )


def _bricks(pygame, surf, w: int, h: int) -> None:
    bw, bh = 16, 8
    for row, y in enumerate(range(0, h, bh)):
        ox = (bw // 2) if row % 2 else 0
        x = -ox
        col = 0
        while x < w:
            body = STONE_LT if (row + col) % 5 == 0 else STONE
            pygame.draw.rect(surf, STONE_DK, (x, y, bw, bh))
            pygame.draw.rect(surf, body, (x + 1, y + 1, bw - 2, bh - 2))
            x += bw
            col += 1


def _sprite(pygame, name: str):
    return pygame.image.load(str(ASSETS / name))


def _banner(pygame, surf, x: int, y: int, w: int, h: int, field) -> None:
    pygame.draw.rect(surf, GOLD_DK, (x + w // 2 - 3, y - 14, 6, 16))
    tip = y + h
    pts = [
        (x, y),
        (x + w, y),
        (x + w - 8, tip - 16),
        (x + w // 2, tip),
        (x + 8, tip - 16),
    ]
    pygame.draw.polygon(surf, field, pts)
    pygame.draw.polygon(surf, GOLD, pts, 3)
    pygame.draw.polygon(surf, GOLD_DK, pts, 1)


def _portrait(pygame, surf, sprite, bx: int, by: int, bw: int, top: int, backing) -> None:
    sw, sh = sprite.get_size()
    x = bx + (bw - sw) // 2
    y = top
    pygame.draw.rect(surf, GOLD_DK, (x - 5, y - 5, sw + 10, sh + 10))
    pygame.draw.rect(surf, backing, (x - 2, y - 2, sw + 4, sh + 4))
    pygame.draw.rect(surf, GOLD, (x - 5, y - 5, sw + 10, sh + 10), 2)
    surf.blit(sprite, (x, y))


def _paddle(pygame, surf, x: int, y: int, w: int, h: int, *, commit: bool, human: bool) -> None:
    field = CRIMSON if human else PARCHMENT
    trim = GOLD
    pygame.draw.rect(surf, GOLD_DK, (x - 3, y - 4, w + 6, h + 8))
    pygame.draw.rect(surf, field, (x, y, w, h))
    pygame.draw.rect(surf, trim, (x, y, w, h), 2)
    stripe = CREAM if human else GOLD_DK
    pygame.draw.rect(surf, stripe, (x + 3, y + 8, max(w - 6, 1), 6))
    if commit:
        pygame.draw.rect(surf, HIGHLIGHT, (x - 4, y - 5, w + 8, h + 10), 2)


def _ball(pygame, surf, cx: int, cy: int, r: int) -> None:
    pygame.draw.circle(surf, GOLD_DK, (cx, cy), r)
    pygame.draw.circle(surf, GOLD, (cx, cy), max(r - 2, 2))
    pygame.draw.circle(surf, HIGHLIGHT, (cx - r // 3, cy - r // 3), max(r // 4, 2))


def _button(pygame, surf, rect, label: str, *, hover: bool) -> None:
    x, y, w, h = rect
    field = GOLD if hover else CRIMSON_DK
    pygame.draw.rect(surf, GOLD_DK, (x - 2, y - 2, w + 4, h + 4))
    pygame.draw.rect(surf, field, (x, y, w, h))
    pygame.draw.rect(surf, GOLD, (x, y, w, h), 2)
    _title, mid, _small = _fonts(pygame)
    text = mid.render(label, True, CREAM)
    surf.blit(text, (x + (w - text.get_width()) // 2, y + (h - text.get_height()) // 2))


def button_rects(fw: int, fh: int) -> dict[str, tuple[int, int, int, int]]:
    bw, bh = 280, 48
    cx = fw // 2 - bw // 2
    cy = fh // 2 - 20
    return {
        "start": (cx, cy, bw, bh),
        "scores": (cx, cy + 64, bw, bh),
        "quit": (cx, cy + 128, bw, bh),
        "back": (cx, fh - 90, bw, bh),
    }


def play_button_rects(fw: int, fh: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "reset": (fw // 2 - 200, fh - 26, 100, 22),
        "menu": (fw // 2 + 100, fh - 26, 100, 22),
    }


def hit(rect: tuple[int, int, int, int], pos: tuple[int, int]) -> bool:
    x, y, w, h = rect
    px, py = pos
    return x <= px <= x + w and y <= py <= y + h


def draw(
    surf,
    state: dict[str, Any],
    C: dict[str, Any],
    hud: dict[str, Any] | None = None,
) -> None:
    pygame = __import__("pygame")
    hud = hud or {}
    fw, fh = frame_size(C)
    ox, s = SIDE, SCALE
    surf.fill(OAK)
    _bricks(pygame, surf, fw, fh)
    beam = 28
    pygame.draw.rect(surf, BEAM, (0, 0, fw, beam))
    pygame.draw.rect(surf, BEAM, (0, fh - beam, fw, beam))
    pygame.draw.rect(surf, GOLD, (0, beam - 3, fw, 3))
    pygame.draw.rect(surf, GOLD, (0, fh - beam, fw, 3))

    bw, bh = 104, 250
    left_b = 8
    right_b = fw - 8 - bw
    _banner(pygame, surf, left_b, beam + 8, bw, bh, CRIMSON)
    _banner(pygame, surf, right_b, beam + 8, bw, bh, PARCHMENT)
    knight = _sprite(pygame, "knight.png")
    fly = _sprite(pygame, "fly.png")
    flw, flh = fly.get_size()
    fly = pygame.transform.scale(fly, (max(flw, 80), max(int(flh * 80 / max(flw, 1)), 64)))
    _portrait(pygame, surf, knight, left_b, beam + 8, bw, beam + 22, CRIMSON)
    _portrait(pygame, surf, fly, right_b, beam + 8, bw, beam + 36, PARCHMENT)

    mid = ox + int(C["width"]) * s // 2
    for y in range(beam + 8, fh - beam - 8, 16):
        pygame.draw.rect(surf, GOLD_DK, (mid - 1, y, 3, 9))
    pygame.draw.circle(surf, CRIMSON_DK, (mid, fh // 2), 16)
    pygame.draw.circle(surf, GOLD, (mid, fh // 2), 16, 3)
    pygame.draw.circle(surf, GOLD, (mid, fh // 2), 5)

    def P(x: float, y: float) -> tuple[int, int]:
        return int(ox + x * s), int(y * s)

    pw = int(C["paddleW"]) * s
    ph = int(C["paddleH"]) * s
    ax, ay = P(float(C["agentX"]), float(state["agent_y"]))
    rx, ry = P(float(C["oppX"]), float(state["opp_y"]))
    _paddle(pygame, surf, ax, ay, pw, ph, commit=False, human=True)
    _paddle(pygame, surf, rx, ry, pw, ph, commit=bool(hud.get("commit")), human=False)
    bx, by = P(float(state["ball_x"]), float(state["ball_y"]))
    _ball(pygame, surf, bx, by, int(C["ballR"]) * s)

    title, _mid, small = _fonts(pygame)
    left = str(hud.get("left_name") or "HUMAN")
    right = str(hud.get("right_name") or "FLY")
    score = f"{left}  {int(state['agent_score'])}     {int(state['opp_score'])}  {right}"
    label = title.render(score, True, CREAM)
    surf.blit(label, (mid - label.get_width() // 2, 4))
    if hud.get("show_play_buttons", True):
        mouse = hud.get("mouse") or (-1, -1)
        for key, rect in play_button_rects(fw, fh).items():
            _button(pygame, surf, rect, key.upper(), hover=hit(rect, mouse))
    cap = small.render(str(hud.get("caption") or "human vs fly"), True, GOLD)
    surf.blit(cap, (mid - cap.get_width() // 2, fh - beam + 6))


def draw_title(surf, C: dict[str, Any], *, mouse: tuple[int, int] = (-1, -1)) -> None:
    pygame = __import__("pygame")
    fw, fh = frame_size(C)
    surf.fill(OAK)
    _bricks(pygame, surf, fw, fh)
    title, mid, small = _fonts(pygame)
    head = title.render("HUMAN VS FLY", True, GOLD)
    surf.blit(head, (fw // 2 - head.get_width() // 2, fh // 2 - 120))
    for key, label in (("start", "START GAME"), ("scores", "HIGH SCORES"), ("quit", "QUIT")):
        rect = button_rects(fw, fh)[key]
        _button(pygame, surf, rect, label, hover=hit(rect, mouse))
    hint = small.render("W/S or mouse  ·  R reset  ·  Esc menu  ·  M mute", True, GOLD_DK)
    surf.blit(hint, (fw // 2 - hint.get_width() // 2, fh - 48))


def draw_scores(
    surf,
    C: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    mouse: tuple[int, int] = (-1, -1),
) -> None:
    pygame = __import__("pygame")
    fw, fh = frame_size(C)
    surf.fill(OAK)
    _bricks(pygame, surf, fw, fh)
    title, mid, small = _fonts(pygame)
    head = title.render("HIGH SCORES", True, GOLD)
    surf.blit(head, (fw // 2 - head.get_width() // 2, 80))
    if not rows:
        empty = mid.render("no matches yet", True, CREAM)
        surf.blit(empty, (fw // 2 - empty.get_width() // 2, fh // 2 - 20))
    else:
        y = 150
        for i, row in enumerate(rows[:10]):
            mark = "W" if row["win"] else "L"
            line = f"{i + 1:2d}   {mark}   HUMAN {row['human']:2d}    FLY {row['fly']:2d}"
            img = small.render(line, True, CREAM)
            surf.blit(img, (fw // 2 - img.get_width() // 2, y))
            y += 28
    _button(pygame, surf, button_rects(fw, fh)["back"], "BACK", hover=hit(button_rects(fw, fh)["back"], mouse))


def rgb_array(state: dict[str, Any], C: dict[str, Any], hud: dict[str, Any] | None = None) -> np.ndarray:
    pygame = __import__("pygame")
    if not pygame.get_init():
        pygame.init()
    fw, fh = frame_size(C)
    surf = pygame.Surface((fw, fh))
    draw(surf, state, C, hud)
    return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()

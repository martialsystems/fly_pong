"""Medieval hall for the Python court. Physics size unchanged. Not a fly title."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SCALE = 2
SIDE = 120
ASSETS = Path(__file__).resolve().parent / "assets"

OAK = (16, 10, 7)
STONE = (58, 45, 34)
STONE_LT = (86, 68, 52)
BEAM = (42, 30, 20)
GOLD = (198, 156, 56)
GOLD_DK = (118, 86, 28)
CRIMSON = (138, 26, 26)
CRIMSON_DK = (78, 14, 14)
CREAM = (230, 210, 168)
IRON = (36, 34, 32)
IRON_LT = (96, 90, 82)
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
    return pygame.font.Font(None, 36), pygame.font.Font(None, 22)


def _bricks(pygame, surf, w: int, h: int) -> None:
    bw, bh = 28 * SCALE, 12 * SCALE
    for row, y in enumerate(range(0, h, bh)):
        ox = (bw // 2) if row % 2 else 0
        x = -ox
        while x < w:
            pygame.draw.rect(surf, STONE, (x + 1, y + 1, bw - 2, bh - 2))
            pygame.draw.rect(surf, STONE_LT, (x + 1, y + 1, bw - 2, 3))
            x += bw


def _sprite(pygame, name: str):
    path = ASSETS / name
    return pygame.image.load(str(path))


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
    pygame.draw.polygon(surf, GOLD, pts, 2)


def _paddle(pygame, surf, x: int, y: int, w: int, h: int, *, commit: bool, human: bool) -> None:
    trim = GOLD if commit or human else IRON_LT
    body = WOOD if human else (72, 48, 28)
    pygame.draw.rect(surf, WOOD_DK, (x - 2, y - 3, w + 4, h + 6))
    pygame.draw.rect(surf, body, (x, y, w, h))
    pygame.draw.rect(surf, trim, (x, y, w, h), 2)
    for fy in (0.2, 0.5, 0.8):
        cy = int(y + h * fy)
        pygame.draw.circle(surf, IRON, (x + w // 2, cy), 3)
        pygame.draw.circle(surf, IRON_LT, (x + w // 2 - 1, cy - 1), 1)
    if human:
        pygame.draw.rect(surf, CRIMSON, (x + 2, y + 6, max(w - 4, 1), 8))
    if commit:
        pygame.draw.rect(surf, GOLD, (x - 3, y - 4, w + 6, h + 8), 3)


def _ball(pygame, surf, cx: int, cy: int, r: int) -> None:
    pygame.draw.circle(surf, IRON, (cx, cy), r)
    pygame.draw.circle(surf, IRON_LT, (cx, cy), r, 2)
    pygame.draw.circle(surf, HIGHLIGHT, (cx - r // 3, cy - r // 3), max(r // 3, 2))


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
    beam = 16 * s // 2 + 8
    pygame.draw.rect(surf, BEAM, (0, 0, fw, beam))
    pygame.draw.rect(surf, BEAM, (0, fh - beam, fw, beam))
    pygame.draw.rect(surf, GOLD_DK, (0, beam - 3, fw, 3))
    pygame.draw.rect(surf, GOLD_DK, (0, fh - beam, fw, 3))

    bw, bh = 104, 250
    left_b = 8
    right_b = fw - 8 - bw
    _banner(pygame, surf, left_b, beam + 8, bw, bh, CRIMSON)
    _banner(pygame, surf, right_b, beam + 8, bw, bh, PARCHMENT)
    knight = _sprite(pygame, "knight.png")
    fly = _sprite(pygame, "fly.png")
    kw, _kh = knight.get_size()
    flw, _flh = fly.get_size()
    surf.blit(knight, (left_b + (bw - kw) // 2, beam + 22))
    surf.blit(fly, (right_b + (bw - flw) // 2, beam + 36))

    mid = ox + int(C["width"]) * s // 2
    for y in range(beam + 8, fh - beam - 8, 16):
        pygame.draw.rect(surf, GOLD_DK, (mid - 1, y, 3, 9))
    pygame.draw.circle(surf, CRIMSON_DK, (mid, fh // 2), 18)
    pygame.draw.circle(surf, GOLD, (mid, fh // 2), 18, 3)
    pygame.draw.circle(surf, GOLD, (mid, fh // 2), 6)
    pygame.draw.circle(surf, GOLD, (ox + 40, beam // 2), 5)
    pygame.draw.circle(surf, GOLD, (fw - ox - 40, beam // 2), 5)

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

    font, small = _fonts(pygame)
    left = str(hud.get("left_name") or "HUMAN")
    right = str(hud.get("right_name") or "FLY")
    score = f"{left}  {int(state['agent_score'])}     {int(state['opp_score'])}  {right}"
    label = font.render(score, True, CREAM)
    surf.blit(label, (mid - label.get_width() // 2, beam + 4))
    caption = str(hud.get("caption") or "human vs fly")
    cap = small.render(caption, True, GOLD)
    surf.blit(cap, (mid - cap.get_width() // 2, fh - beam + 6))


def rgb_array(state: dict[str, Any], C: dict[str, Any], hud: dict[str, Any] | None = None) -> np.ndarray:
    pygame = __import__("pygame")
    if not pygame.get_init():
        pygame.init()
    fw, fh = frame_size(C)
    surf = pygame.Surface((fw, fh))
    draw(surf, state, C, hud)
    return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()

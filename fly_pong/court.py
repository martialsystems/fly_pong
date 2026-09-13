"""Medieval hall for the Python court. Same physics. Not a fly title."""

from __future__ import annotations

from typing import Any

import numpy as np

OAK = (16, 10, 7)
STONE = (58, 45, 34)
STONE_LT = (86, 68, 52)
MORTAR = (32, 24, 18)
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
SABLE = (14, 12, 10)
HIGHLIGHT = (240, 228, 196)

def _fonts(pygame):
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame.font.Font(None, 24), pygame.font.Font(None, 16)


def _bricks(pygame, surf, w: int, h: int) -> None:
    bw, bh = 28, 12
    for row, y in enumerate(range(0, h, bh)):
        ox = (bw // 2) if row % 2 else 0
        x = -ox
        while x < w:
            pygame.draw.rect(surf, STONE, (x + 1, y + 1, bw - 2, bh - 2))
            pygame.draw.rect(surf, STONE_LT, (x + 1, y + 1, bw - 2, 2))
            x += bw


def _banner(pygame, surf, x: int, field, stripe) -> None:
    pygame.draw.rect(surf, GOLD_DK, (x + 10, 18, 4, 10))
    pts = [(x, 28), (x + 24, 28), (x + 20, 108), (x + 12, 118), (x + 4, 108)]
    pygame.draw.polygon(surf, field, pts)
    pygame.draw.polygon(surf, GOLD, pts, 1)
    pygame.draw.rect(surf, stripe, (x + 10, 32, 4, 70))


def _paddle(pygame, surf, x: int, y: int, w: int, h: int, *, commit: bool, you: bool) -> None:
    trim = GOLD if commit or you else IRON_LT
    body = WOOD if you else (72, 48, 28)
    dark = WOOD_DK
    pygame.draw.rect(surf, dark, (x - 1, y - 2, w + 2, h + 4))
    pygame.draw.rect(surf, body, (x, y, w, h))
    pygame.draw.rect(surf, trim, (x, y, w, h), 1)
    for i, fy in enumerate((0.2, 0.5, 0.8)):
        cy = int(y + h * fy)
        pygame.draw.circle(surf, IRON, (x + w // 2, cy), 2)
        pygame.draw.circle(surf, IRON_LT, (x + w // 2 - 1, cy - 1), 1)
    if you:
        pygame.draw.rect(surf, CRIMSON, (x + 1, y + 4, max(w - 2, 1), 6))
    if commit:
        pygame.draw.rect(surf, GOLD, (x - 2, y - 3, w + 4, h + 6), 2)


def _ball(pygame, surf, cx: int, cy: int, r: int) -> None:
    pygame.draw.circle(surf, IRON, (cx, cy), r)
    pygame.draw.circle(surf, IRON_LT, (cx, cy), r, 1)
    pygame.draw.circle(surf, HIGHLIGHT, (cx - r // 3, cy - r // 3), max(r // 3, 1))


def draw(
    surf,
    state: dict[str, Any],
    C: dict[str, Any],
    hud: dict[str, Any] | None = None,
) -> None:
    pygame = __import__("pygame")
    hud = hud or {}
    w = int(C["width"])
    h = int(C["height"])
    surf.fill(OAK)
    _bricks(pygame, surf, w, h)
    pygame.draw.rect(surf, BEAM, (0, 0, w, 18))
    pygame.draw.rect(surf, BEAM, (0, h - 18, w, 18))
    pygame.draw.rect(surf, GOLD_DK, (0, 16, w, 2))
    pygame.draw.rect(surf, GOLD_DK, (0, h - 18, w, 2))
    _banner(pygame, surf, 6, CRIMSON, GOLD)
    _banner(pygame, surf, w - 30, SABLE, GOLD)
    mid = w // 2
    for y in range(22, h - 22, 14):
        pygame.draw.rect(surf, GOLD_DK, (mid - 1, y, 2, 8))
    pygame.draw.circle(surf, CRIMSON_DK, (mid, h // 2), 14)
    pygame.draw.circle(surf, GOLD, (mid, h // 2), 14, 2)
    pygame.draw.circle(surf, GOLD, (mid, h // 2), 5)
    pygame.draw.circle(surf, GOLD, (36, 9), 4)
    pygame.draw.circle(surf, GOLD, (w - 36, 9), 4)

    pw = int(C["paddleW"])
    ph = int(C["paddleH"])
    _paddle(
        pygame,
        surf,
        int(C["agentX"]),
        int(state["agent_y"]),
        pw,
        ph,
        commit=False,
        you=True,
    )
    _paddle(
        pygame,
        surf,
        int(C["oppX"]),
        int(state["opp_y"]),
        pw,
        ph,
        commit=bool(hud.get("commit")),
        you=False,
    )
    _ball(pygame, surf, int(state["ball_x"]), int(state["ball_y"]), int(C["ballR"]))

    font, small = _fonts(pygame)
    left = str(hud.get("left_name") or "YOU")
    right = str(hud.get("right_name") or "GOALIE")
    score = f"{left}  {int(state['agent_score'])}     {int(state['opp_score'])}  {right}"
    label = font.render(score, True, CREAM)
    surf.blit(label, (mid - label.get_width() // 2, 20))
    caption = str(hud.get("caption") or "human vs approach_commit")
    cap = small.render(caption, True, GOLD)
    surf.blit(cap, (mid - cap.get_width() // 2, h - 16))


def rgb_array(state: dict[str, Any], C: dict[str, Any], hud: dict[str, Any] | None = None) -> np.ndarray:
    pygame = __import__("pygame")
    if not pygame.get_init():
        pygame.init()
    w = int(C["width"])
    h = int(C["height"])
    surf = pygame.Surface((w, h))
    draw(surf, state, C, hud)
    return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()

#!/usr/bin/env python3
"""Downsample painted knight/fly stills into small-pixel banner sprites."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fly_pong" / "assets"
KNIGHT_IN = OUT / "src" / "knight_paint.jpg"
FLY_IN = OUT / "src" / "fly_paint.jpg"


def _load(path: Path) -> np.ndarray:
    surf = pygame.image.load(str(path))
    arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
    return arr.astype(np.uint8)


def _key_cream(rgb: np.ndarray, thresh: int = 28) -> np.ndarray:
    """RGBA with pale cream keyed out."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    pale = (r > 185) & (g > 175) & (b > 145) & (np.abs(r - g) < thresh) & (np.abs(g - b) < 50)
    a = np.where(pale, 0, 255).astype(np.uint8)
    out = np.dstack([rgb, a])
    return out


def _bbox(rgba: np.ndarray, pad: int = 8) -> np.ndarray:
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 16)
    if xs.size == 0:
        return rgba
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(rgba.shape[1], x1 + pad)
    y1 = min(rgba.shape[0], y1 + pad)
    return rgba[y0:y1, x0:x1]


def _fit(rgba: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Box-downsample to a small pixel grid. That is the pixel tool."""
    h, w = rgba.shape[:2]
    scale = min(max_w / float(w), max_h / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    ys = np.linspace(0, h, nh + 1).astype(int)
    xs = np.linspace(0, w, nw + 1).astype(int)
    out = np.zeros((nh, nw, 4), dtype=np.uint8)
    step = 4
    for j in range(nh):
        for i in range(nw):
            block = rgba[ys[j] : ys[j + 1], xs[i] : xs[i + 1]]
            if block.size == 0:
                continue
            opaque = block[:, :, 3] > 40
            if opaque.mean() < 0.35:
                continue
            rgb = block[opaque][:, :3].mean(axis=0)
            out[j, i, :3] = (rgb.astype(np.uint16) // step) * step
            out[j, i, 3] = 255
    return out


def _save(rgba: np.ndarray, path: Path) -> None:
    pygame.surfarray.make_surface  # keep import used
    h, w = rgba.shape[:2]
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    px = pygame.surfarray.pixels3d(surf)
    px[:, :, :] = np.transpose(rgba[:, :, :3], (1, 0, 2))
    del px
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[:, :] = np.transpose(rgba[:, :, 3], (1, 0))
    del pa
    pygame.image.save(surf, str(path))
    print(f"wrote {path.relative_to(ROOT)} {w}x{h}")


def main() -> None:
    pygame.init()
    OUT.mkdir(parents=True, exist_ok=True)
    knight = _fit(_bbox(_key_cream(_load(KNIGHT_IN))), 80, 104)
    fly = _fit(_bbox(_key_cream(_load(FLY_IN), thresh=36)), 88, 64)
    _save(knight, OUT / "knight.png")
    _save(fly, OUT / "fly.png")
    pygame.quit()


if __name__ == "__main__":
    main()

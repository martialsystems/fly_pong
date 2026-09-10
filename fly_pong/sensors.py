"""Game state to a 1-D vertical luminance strip (ommatidia)."""

from __future__ import annotations

from typing import Any

import numpy as np


class CompoundEye:
    """Bright ball on a dark field, sampled along the vertical retina."""

    def __init__(self, n_ommatidia: int = 32, sigma: float = 1.6):
        self.n = int(n_ommatidia)
        self.sigma = float(sigma)
        self.ys = np.linspace(0.0, 1.0, self.n, dtype=np.float32)

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        height = float(state["height"])
        ball_y = float(state["ball_y"]) / height
        d = (self.ys - ball_y) * self.n
        field = np.exp(-0.5 * (d / self.sigma) ** 2)
        return field.astype(np.float32)

    def com(self, field: np.ndarray) -> float:
        """Center of mass along the vertical retina, in [0, 1] (0 is the top)."""
        luma = np.asarray(field, dtype=np.float32)
        mass = float(luma.sum())
        if mass <= 1e-8:
            return 0.5
        return float(np.dot(luma, self.ys) / mass)

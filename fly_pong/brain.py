"""Reichardt / Barlow-Levick T4/T5 array. Not a connectome execution."""

from __future__ import annotations

import numpy as np


class T4T5MotionCircuit:
    """
    Luminance to ON/OFF half-wave to delayed neighbor product to T4/T5 to motor.

    T4: ON-edge (L1). T5: OFF-edge (L2).
    Subtypes used here: c up, d down. Pong only needs vertical motion.
    """

    def __init__(self, n: int = 32, tau: float = 0.35, dt: float = 1.0):
        self.n = int(n)
        self.alpha = float(np.exp(-dt / tau))
        self.prev_luma = np.zeros(self.n, dtype=np.float32)
        self.delay_on = np.zeros(self.n, dtype=np.float32)
        self.delay_off = np.zeros(self.n, dtype=np.float32)

    def reset(self) -> None:
        self.prev_luma[:] = 0
        self.delay_on[:] = 0
        self.delay_off[:] = 0

    def step(self, luma: np.ndarray) -> dict[str, float]:
        luma = np.asarray(luma, dtype=np.float32)
        if luma.shape != (self.n,):
            raise ValueError(f"luma shape {luma.shape} != ({self.n},)")
        dI = luma - self.prev_luma
        on = np.clip(dI, 0, None)
        off = np.clip(-dI, 0, None)

        self.delay_on = self.alpha * self.delay_on + (1.0 - self.alpha) * on
        self.delay_off = self.alpha * self.delay_off + (1.0 - self.alpha) * off

        # ys increase downward on the court. Neighbor i → i+1 is screen-down.
        t4_down = self.delay_on[:-1] * on[1:]
        t4_up = self.delay_on[1:] * on[:-1]
        t5_down = self.delay_off[:-1] * off[1:]
        t5_up = self.delay_off[1:] * off[:-1]

        vs_up = float(t4_up.mean() + t5_up.mean())
        vs_down = float(t4_down.mean() + t5_down.mean())
        self.prev_luma = luma.copy()
        return {
            "T4c": float(t4_up.mean()),
            "T4d": float(t4_down.mean()),
            "T5c": float(t5_up.mean()),
            "T5d": float(t5_down.mean()),
            "VS_up": vs_up,
            "VS_down": vs_down,
            "steering": vs_down - vs_up,
        }


def decode_motor(steering: float, threshold: float = 1e-5) -> int:
    """Firing-rate imbalance to Gym action: 0 stay, 1 up, 2 down."""
    if steering > threshold:
        return 2
    if steering < -threshold:
        return 1
    return 0

"""Reichardt / Barlow-Levick T4/T5 array. Not a connectome execution."""

from __future__ import annotations

from collections import deque

import numpy as np

from fly_pong.constants import load_constants

# Correlator delay. 1.5 frames ≈ 25 ms at fps 60. Mi1/Tm3-sized. dt=1.
T4T5_TAU = 1.5
# Judgment FIFO. Placeholder until clocks live in the neurons.
# delay_ms_assumed = MOTOR_DELAY_FRAMES * 1000 / fps
MOTOR_DELAY_FRAMES = 2  # 33 ms at 60 fps. 1≈17 ms, 3≈50 ms. Stop at 3.


def delay_ms_assumed(frames: int | None = None, fps: float | None = None) -> float:
    rate = float(load_constants()["fps"] if fps is None else fps)
    n = MOTOR_DELAY_FRAMES if frames is None else int(frames)
    return n * 1000.0 / rate


class MotorDelay:
    """Fixed FIFO of paddle dy. Read the oldest sample.

    Centering stays instant inside the circuit. This is the late paddle.
    """

    def __init__(self, frames: int | None = None):
        C = load_constants()
        self.n = int(MOTOR_DELAY_FRAMES if frames is None else frames)
        self.fps = float(C["fps"])
        # delay_ms_assumed = MOTOR_DELAY_FRAMES * 1000 / fps
        self.delay_ms_assumed = self.n * 1000.0 / self.fps
        self.reset()

    def reset(self) -> None:
        if self.n <= 0:
            self._q = deque()
            return
        self._q = deque([0.0] * self.n, maxlen=self.n)

    def push(self, dy: float) -> float:
        if self.n <= 0:
            return float(dy)
        out = float(self._q[0])
        self._q.append(float(dy))
        return out


class T4T5MotionCircuit:
    """
    Luminance to ON/OFF half-wave to delayed neighbor product to T4/T5 to motor.

    T4: ON-edge (L1). T5: OFF-edge (L2).
    Subtypes used here: c up, d down. Pong only needs vertical motion.
    """

    def __init__(self, n: int = 32, tau: float = T4T5_TAU, dt: float = 1.0):
        self.n = int(n)
        self.tau = float(tau)
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

"""Pong state to T4/T5 to Discrete(3). Swap the circuit, keep this interface."""

from __future__ import annotations

from typing import Any

import numpy as np

from fly_pong.brain import T4T5MotionCircuit, decode_motor
from fly_pong.sensors import CompoundEye


class FlyBrainBridge:
    """[Pong state] → retinotopic stimulus → T4/T5 → paddle command."""

    def __init__(self, n_ommatidia: int = 32):
        self.eye = CompoundEye(n_ommatidia)
        self.circuit = T4T5MotionCircuit(n=n_ommatidia)

    def reset(self) -> None:
        self.circuit.reset()

    def translate_input(self, state: dict[str, Any]) -> np.ndarray:
        return self.eye.encode(state)

    def brain_step(self, state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        stimulus = self.translate_input(state)
        rates = self.circuit.step(stimulus)
        action = decode_motor(rates["steering"])
        return action, {"stimulus": stimulus, **rates}

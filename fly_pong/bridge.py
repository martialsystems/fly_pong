"""Pong state to T4/T5 to Discrete(3). Swap the circuit, keep this interface."""

from __future__ import annotations

from typing import Any

import numpy as np

from fly_pong.brain import T4T5MotionCircuit, decode_motor
from fly_pong.sensors import CompoundEye


class FlyBrainBridge:
    """Pong state to retinotopic field to T4/T5 plus a centering reflex.

    Motion-only T4/T5 scored 0/20 matches against the lag paddle: a still or
    horizontally moving ball produces almost no vertical edge motion, so the
    paddle never acquires the blob. Centering compares the luminance center of
    mass to the paddle's vertical locus (proprioception). T4/T5 still adds a
    small lead. Gain is higher when the ball's horizontal velocity is toward
    the agent (an approach / looming-style boost).
    """

    def __init__(
        self,
        n_ommatidia: int = 32,
        vel_gain: float = 40.0,
        pos_threshold: float = 0.02,
        approach_gain: float = 1.4,
        recede_gain: float = 0.85,
    ):
        self.eye = CompoundEye(n_ommatidia)
        self.circuit = T4T5MotionCircuit(n=n_ommatidia)
        self.vel_gain = float(vel_gain)
        self.pos_threshold = float(pos_threshold)
        self.approach_gain = float(approach_gain)
        self.recede_gain = float(recede_gain)

    def reset(self) -> None:
        self.circuit.reset()

    def translate_input(self, state: dict[str, Any]) -> np.ndarray:
        return self.eye.encode(state)

    def brain_step(self, state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        stimulus = self.translate_input(state)
        rates = self.circuit.step(stimulus)
        com = self.eye.com(stimulus)
        paddle_h = float(state["paddle_h"])
        height = float(state["height"])
        paddle_c = (float(state["paddle_y"]) + paddle_h / 2.0) / height
        pos_err = com - paddle_c
        toward_agent = float(state.get("ball_vx", 0.0)) < 0.0
        gain = self.approach_gain if toward_agent else self.recede_gain
        command = gain * pos_err + self.vel_gain * float(rates["steering"])
        action = decode_motor(command, threshold=self.pos_threshold)
        return action, {
            "stimulus": stimulus,
            **rates,
            "com": com,
            "pos_err": pos_err,
            "command": command,
            "approach": toward_agent,
        }

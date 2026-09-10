"""Named move/aim channels. Frozen filters, not a connectome query."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fly_pong.brain import T4T5MotionCircuit
from fly_pong.constants import load_constants
from fly_pong.physics import clip
from fly_pong.sensors import CompoundEye

MOVE_KEYS = (
    "T4c",
    "T4d",
    "T5c",
    "T5d",
    "VS_up",
    "VS_down",
    "error_y",
    "ball_vy",
    "time_to_own_goal",
)
AIM_KEYS = (
    "lc_blob",
    "predicted_contact_y",
    "predicted_contact_t",
    "opponent_y",
    "opponent_open_up",
    "opponent_open_down",
    "desired_offset",
)

RATE_SCALE = 80.0
TIME_CAP = 120.0
AIM_N = 10


@dataclass
class FeatureBank:
    move: np.ndarray
    aim: np.ndarray
    names_move: tuple[str, ...]
    names_aim: tuple[str, ...]
    rates: dict[str, float]
    frames_to_paddle: float
    predicted_contact_y_px: float
    incoming: bool


def _paddle_y(state: dict[str, Any]) -> float:
    if "paddle_y" in state:
        return float(state["paddle_y"])
    return float(state["agent_y"])


def _paddle_h(state: dict[str, Any], C: dict[str, Any]) -> float:
    if "paddle_h" in state:
        return float(state["paddle_h"])
    return float(C["paddleH"])


def frames_to_agent_paddle(state: dict[str, Any], C: dict[str, Any] | None = None) -> float:
    C = C or load_constants()
    vx = float(state["ball_vx"])
    if vx >= -1e-6:
        return TIME_CAP
    edge = float(C["agentX"]) + float(C["paddleW"]) + float(C["ballR"])
    dist = float(state["ball_x"]) - edge
    if dist <= 0:
        return 0.0
    return float(min(TIME_CAP, dist / max(abs(vx), 1e-6)))


def desired_offset(state: dict[str, Any], C: dict[str, Any] | None = None) -> float:
    """Positive offset sends the ball down the court (larger y)."""
    C = C or load_constants()
    h = float(C["height"])
    ph = _paddle_h(state, C)
    opp_y = float(state["opp_y"])
    open_up = opp_y / h
    open_down = (h - (opp_y + ph)) / h
    return float(clip(open_down - open_up, -1.0, 1.0))


class FeatureEncoder:
    def __init__(self, n_ommatidia: int = 32):
        self.eye = CompoundEye(n_ommatidia)
        self.circuit = T4T5MotionCircuit(n=n_ommatidia)

    def reset(self) -> None:
        self.circuit.reset()

    def encode(self, state: dict[str, Any]) -> FeatureBank:
        C = load_constants()
        h = float(state.get("height") or C["height"])
        ph = _paddle_h(state, C)
        py = _paddle_y(state)
        luma = self.eye.encode(
            {"height": h, "ball_y": float(state["ball_y"]), "ball_x": float(state["ball_x"])}
        )
        rates = self.circuit.step(luma)
        com = self.eye.com(luma)
        paddle_c = (py + ph / 2.0) / h
        error_y = com - paddle_c
        t_hit = frames_to_agent_paddle(state, C)
        incoming = float(state["ball_vx"]) < 0.0
        pred_y_px = float(state["ball_y"]) + float(state["ball_vy"]) * t_hit
        pred_y_px = clip(pred_y_px, float(C["ballR"]), h - float(C["ballR"]))
        opp_y = float(state["opp_y"])
        open_up = opp_y / h
        open_down = (h - (opp_y + ph)) / h
        scale = float(C["velScale"])
        move = np.array(
            [
                rates["T4c"] * RATE_SCALE,
                rates["T4d"] * RATE_SCALE,
                rates["T5c"] * RATE_SCALE,
                rates["T5d"] * RATE_SCALE,
                rates["VS_up"] * RATE_SCALE,
                rates["VS_down"] * RATE_SCALE,
                error_y,
                clip(float(state["ball_vy"]) / scale, -1.0, 1.0),
                t_hit / TIME_CAP if incoming else 1.0,
            ],
            dtype=np.float32,
        )
        aim = np.array(
            [
                float(luma.max()),
                pred_y_px / h,
                t_hit / TIME_CAP,
                (opp_y + ph / 2.0) / h,
                open_up,
                open_down,
                desired_offset(state, C),
            ],
            dtype=np.float32,
        )
        return FeatureBank(
            move=move,
            aim=aim,
            names_move=MOVE_KEYS,
            names_aim=AIM_KEYS,
            rates=rates,
            frames_to_paddle=t_hit,
            predicted_contact_y_px=pred_y_px,
            incoming=incoming,
        )

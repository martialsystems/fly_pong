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
    "opponent_y",
    "opponent_open_up",
    "opponent_open_down",
    "desired_offset",
)
UNUSED_MOVE_KEYS = (
    "HS_close",
    "LPi_opp",
    "LC10_track",
    "LC11_dark",
    "LC18_fine",
    "LPLC_loom",
    "LC4_loom_center",
)

RATE_SCALE = 80.0
TIME_CAP = 120.0
AIM_N = 24
MB_N = 8


@dataclass
class FeatureBank:
    move: np.ndarray
    aim: np.ndarray
    unused: np.ndarray
    names_move: tuple[str, ...]
    names_aim: tuple[str, ...]
    names_unused: tuple[str, ...]
    rates: dict[str, float]
    frames_to_paddle: float
    predicted_contact_y_px: float
    incoming: bool
    cx_heading: float
    mb_value: float


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


def hs_close(state: dict[str, Any], C: dict[str, Any] | None = None) -> float:
    """|vx| toward own goal. Zero when receding. Not paddle dy."""
    C = C or load_constants()
    vx = float(state["ball_vx"])
    if vx >= 0.0:
        return 0.0
    return float(clip(abs(vx) / float(C["velScale"]), 0.0, 1.0))


def lpi_opp(rates: dict[str, float], ball_vy: float) -> float:
    """Vertical PD minus ND from existing T4/T5. Opponent subtract."""
    up = float(rates["T4c"]) + float(rates["T5c"])
    down = float(rates["T4d"]) + float(rates["T5d"])
    if ball_vy >= 0.0:
        raw = down - up
    else:
        raw = up - down
    return float(clip(raw * RATE_SCALE, -1.0, 1.0))


def lc10_track(luma: np.ndarray, state: dict[str, Any], C: dict[str, Any] | None = None) -> float:
    """Small-object energy at the ball, size ~ ball/paddle, moving."""
    C = C or load_constants()
    size = float(C["ballR"]) / float(C["paddleH"])
    moving = abs(float(state["ball_vx"])) + abs(float(state["ball_vy"]))
    motion = clip(moving / float(C["velScale"]), 0.0, 1.0)
    return float(clip(float(np.max(luma)) * size * motion, 0.0, 1.0))


def lc11_dark(luma: np.ndarray) -> float:
    """Small inverted-blob energy on the 1-D strip."""
    field = np.asarray(luma, dtype=np.float32)
    inv = 1.0 - np.clip(field, 0.0, 1.0)
    high = inv - float(inv.mean())
    return float(clip(float(np.max(np.abs(high))), 0.0, 1.0))


def lc18_fine(luma: np.ndarray, prev_luma: np.ndarray, ball_y: float, height: float) -> float:
    """Higher-spatial-frequency motion at the ball bin."""
    field = np.asarray(luma, dtype=np.float32)
    prev = np.asarray(prev_luma, dtype=np.float32)
    n = int(field.shape[0])
    if n < 3:
        return 0.0
    i = int(np.clip(round((float(ball_y) / max(float(height), 1e-6)) * (n - 1)), 1, n - 2))
    spatial = abs(float(field[i] - 0.5 * (field[i - 1] + field[i + 1])))
    temporal = abs(float(field[i] - prev[i])) if prev.shape == field.shape else 0.0
    return float(clip(spatial * temporal * 8.0, 0.0, 1.0))


def lplc_loom(state: dict[str, Any], C: dict[str, Any] | None = None) -> float:
    """tau = dist_x / |vx| when incoming, normalized. Else 1 (far)."""
    C = C or load_constants()
    if float(state["ball_vx"]) >= 0.0:
        return 1.0
    return float(clip(frames_to_agent_paddle(state, C) / TIME_CAP, 0.0, 1.0))


def lc4_loom_center(state: dict[str, Any], C: dict[str, Any] | None = None) -> float:
    """Loom only when incoming and the ball is in own half."""
    C = C or load_constants()
    w = float(state.get("width") or C["width"])
    if float(state["ball_vx"]) >= 0.0:
        return 0.0
    if float(state["ball_x"]) >= 0.5 * w:
        return 0.0
    return lplc_loom(state, C)


def cx_heading(state: dict[str, Any], C: dict[str, Any] | None = None) -> float:
    """Signed open-corner heading in {-1, 0, +1}. Trace only."""
    d = desired_offset(state, C)
    if d > 0.05:
        return 1.0
    if d < -0.05:
        return -1.0
    return 0.0


def unused_move_vector(
    *,
    luma: np.ndarray,
    prev_luma: np.ndarray,
    rates: dict[str, float],
    state: dict[str, Any],
    C: dict[str, Any],
) -> np.ndarray:
    h = float(state.get("height") or C["height"])
    return np.array(
        [
            hs_close(state, C),
            lpi_opp(rates, float(state["ball_vy"])),
            lc10_track(luma, state, C),
            lc11_dark(luma),
            lc18_fine(luma, prev_luma, float(state["ball_y"]), h),
            lplc_loom(state, C),
            lc4_loom_center(state, C),
        ],
        dtype=np.float32,
    )


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
        self._geo_hist: list[float] = []

    def reset(self) -> None:
        self.circuit.reset()
        self._geo_hist = []

    def observe_contact(self, signed_open_geo: float) -> None:
        self._geo_hist.append(float(signed_open_geo))
        if len(self._geo_hist) > MB_N:
            self._geo_hist.pop(0)

    def mb_value(self) -> float:
        if not self._geo_hist:
            return 0.5
        return float(np.mean(self._geo_hist))

    def encode(self, state: dict[str, Any]) -> FeatureBank:
        C = load_constants()
        h = float(state.get("height") or C["height"])
        ph = _paddle_h(state, C)
        py = _paddle_y(state)
        luma = self.eye.encode(
            {"height": h, "ball_y": float(state["ball_y"]), "ball_x": float(state["ball_x"])}
        )
        prev_luma = self.circuit.prev_luma.copy()
        rates = self.circuit.step(luma)
        unused = unused_move_vector(luma=luma, prev_luma=prev_luma, rates=rates, state=state, C=C)
        heading = cx_heading(state, C)
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
            unused=unused,
            names_move=MOVE_KEYS,
            names_aim=AIM_KEYS,
            names_unused=UNUSED_MOVE_KEYS,
            rates=rates,
            frames_to_paddle=t_hit,
            predicted_contact_y_px=pred_y_px,
            incoming=incoming,
            cx_heading=heading,
            mb_value=self.mb_value(),
        )

"""Commit-if-reachable veto. Contact, not placement.

Valence flip: ESCAPE_SIGN = -1 occupies the crossing Y (approach).
ESCAPE_SIGN = +1 flees it. Same 24-frame gate as landing/loom.
Not an aim title and not a fly result.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from fly_pong.bridge import FlyBrainBridge
from fly_pong.constants import load_constants
from fly_pong.features import AIM_KEYS, AIM_N, MOVE_KEYS, frames_to_agent_paddle
from fly_pong.physics import clip, dy_from_action


# +1 flee (leave the point). -1 approach (occupy the crossing Y).
ESCAPE_SIGN = -1
COMMIT_WINDOW = AIM_N  # 24. Do not widen.
K_APPROACH = 1.0
# Saturates at the existing centering deadband (FlyBrainBridge.pos_threshold).
CENTER_DEADBAND = 0.02


def reach_frames(y_pred: float, paddle_center: float, paddle_speed: float) -> float:
    speed = max(float(paddle_speed), 1e-6)
    return abs(float(y_pred) - float(paddle_center)) / speed


def should_commit(
    *,
    incoming: bool,
    tau: float,
    y_pred: float,
    paddle_center: float,
    paddle_speed: float,
    window: float,
) -> bool:
    """Lunge only if the intercept is in range before contact.

    tau: frames to own paddle. window: short time-to-contact (AIM_N).
    Firing when reach > tau is the 7/40 setpoint path. Do not.
    """
    if not incoming:
        return False
    if float(tau) > float(window):
        return False
    return reach_frames(y_pred, paddle_center, paddle_speed) <= float(tau)


def reflected_crossing_y(
    y_ball: float,
    v_y: float,
    t_to_line: float,
    height: float,
    ball_r: float,
) -> float:
    """Crossing Y after t frames. Reflect top/bottom if that bounce is in the window.

    Unfolds the corridor. Not a wall planner.
    """
    lo = float(ball_r)
    hi = float(height) - float(ball_r)
    span = hi - lo
    y = float(y_ball)
    vy = float(v_y)
    t = float(t_to_line)
    if t <= 0.0 or abs(vy) < 1e-12:
        return clip(y, lo, hi)
    if span <= 1e-9:
        return lo
    pos = y - lo
    raw = pos + vy * t
    period = 2.0 * span
    m = raw % period
    if m < 0.0:
        m += period
    if m <= span:
        return m + lo
    return (period - m) + lo


def _paddle_center(state: dict[str, Any], C: dict[str, Any]) -> float:
    if "paddle_y" in state:
        py = float(state["paddle_y"])
    else:
        py = float(state["agent_y"])
    if "paddle_h" in state:
        ph = float(state["paddle_h"])
    else:
        ph = float(C["paddleH"])
    return py + ph / 2.0


def _height(state: dict[str, Any], C: dict[str, Any]) -> float:
    return float(state.get("height") or C["height"])


def approach_commit(
    state: dict[str, Any],
    *,
    oracle_y: bool = False,
    y_est: float | None = None,
    vy_est: float | None = None,
    error_y: float | None = None,
    escape_sign: int | None = None,
    k: float | None = None,
    k_center: float | None = None,
    window: float | None = None,
    paddle_speed: float | None = None,
) -> dict[str, Any]:
    """Paddle velocity from the valence-flip commit gate.

    Default: mild centering (or hold if error_y is missing).
    Incoming and t_to_line ≤ window and reachable:
      dy = clip(ESCAPE_SIGN * k * (y_paddle - y_target))
    ESCAPE_SIGN -1: approach. +1: flee.
    """
    C = load_constants()
    sign = int(ESCAPE_SIGN if escape_sign is None else escape_sign)
    gain = float(K_APPROACH if k is None else k)
    win = float(COMMIT_WINDOW if window is None else window)
    speed = float(C["paddleSpeed"] if paddle_speed is None else paddle_speed)
    gain_c = float(speed / CENTER_DEADBAND if k_center is None else k_center)
    h = _height(state, C)
    r = float(C["ballR"])
    y_paddle = _paddle_center(state, C)
    incoming = float(state["ball_vx"]) < 0.0
    t_to_line = float(frames_to_agent_paddle(state, C))
    in_window = bool(incoming and t_to_line <= win)

    if oracle_y:
        y_ball = float(state["ball_y"])
        v_y = float(state["ball_vy"])
    elif y_est is not None:
        y_ball = float(y_est)
        v_y = 0.0 if vy_est is None else float(vy_est)
    else:
        y_ball = y_paddle
        v_y = 0.0

    y_target = reflected_crossing_y(y_ball, v_y, t_to_line, h, r)
    reach = reach_frames(y_target, y_paddle, speed)
    reachable = bool(reach <= t_to_line)
    commit = bool(in_window and reachable)
    reach_miss = bool(in_window and not reachable)
    abs_err = abs(y_target - y_paddle)
    err_norm = abs_err / max(h, 1e-6)

    if commit:
        dy = clip(sign * gain * (y_paddle - y_target), -speed, speed)
    elif error_y is not None:
        dy = clip(gain_c * float(error_y), -speed, speed)
    else:
        dy = 0.0

    action = 0
    if dy < -0.5:
        action = 1
    elif dy > 0.5:
        action = 2

    return {
        "dy": float(dy),
        "action": int(action),
        "commit": commit,
        "in_window": in_window,
        "incoming": incoming,
        "reachable": reachable,
        "reach_miss": reach_miss,
        "reach": float(reach),
        "t_to_line": t_to_line,
        "tau": t_to_line,
        "y_target": float(y_target),
        "y_ball": float(y_ball),
        "v_y": float(v_y),
        "y_paddle": float(y_paddle),
        "abs_error_y": float(err_norm),
        "escape_sign": sign,
        "oracle_y": bool(oracle_y),
        "window": win,
    }


class ApproachCommitController:
    """Hand loop (strip → T4/T5 → centering) plus last-second motor remap.

    play_match-compatible. Offset stays off.
    """

    def __init__(
        self,
        *,
        oracle_y: bool = False,
        escape_sign: int | None = None,
        n_ommatidia: int = 32,
        window: float | None = None,
    ):
        self.bridge = FlyBrainBridge(n_ommatidia=n_ommatidia)
        self.oracle_y = bool(oracle_y)
        self.escape_sign = int(ESCAPE_SIGN if escape_sign is None else escape_sign)
        self.window = float(COMMIT_WINDOW if window is None else window)
        self.aim_n = int(self.window)
        self.C = load_constants()
        self.encoder = SimpleNamespace(observe_contact=lambda *_a, **_k: None)
        self._prev_com: float | None = None
        self.stats: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self.bridge.reset()
        self._prev_com = None
        self.stats = {
            "commit_frames": 0,
            "window_frames": 0,
            "reach_misses": 0,
            "total_frames": 0,
            "abs_error_at_commit": [],
        }

    def step_command(self, state: dict[str, Any]) -> dict[str, Any]:
        action_c, neural = self.bridge.brain_step(state)
        com = float(neural["com"])
        h = _height(state, self.C)
        y_est = com * h
        if self._prev_com is None:
            vy_est = 0.0
        else:
            vy_est = (com - self._prev_com) * h
        self._prev_com = com

        cmd = approach_commit(
            state,
            oracle_y=self.oracle_y,
            y_est=y_est,
            vy_est=vy_est,
            error_y=float(neural["pos_err"]),
            escape_sign=self.escape_sign,
            window=self.window,
        )
        if cmd["commit"]:
            dy = float(cmd["dy"])
        else:
            dy = float(dy_from_action(int(action_c), C=self.C))
        dy = float(self.bridge.motor.push(dy))
        cmd["dy"] = dy
        action_c = 0
        if dy < -0.5:
            action_c = 1
        elif dy > 0.5:
            action_c = 2
        cmd["action"] = action_c

        self.stats["total_frames"] += 1
        if cmd["in_window"]:
            self.stats["window_frames"] += 1
        if cmd["commit"]:
            self.stats["commit_frames"] += 1
            self.stats["abs_error_at_commit"].append(float(cmd["abs_error_y"]))
        if cmd["reach_miss"]:
            self.stats["reach_misses"] += 1

        action = int(cmd["action"])
        return {
            "dy": dy,
            "action": action,
            "u_dy": 0.0,
            "u_offset": 0.0,
            "u_offset_head": 0.0,
            "aim_active": bool(cmd["commit"]),
            "commit": bool(cmd["commit"]),
            "in_window": bool(cmd["in_window"]),
            "reach": float(cmd["reach"]),
            "tau": float(cmd["t_to_line"]),
            "target_center_px": float(cmd["y_target"]),
            "paddle_center_px": float(cmd["y_paddle"]),
            "g_move": np.zeros(len(MOVE_KEYS), dtype=np.float32),
            "g_aim": np.zeros(len(AIM_KEYS), dtype=np.float32),
            "cx_heading": 0.0,
            "mb_value": 0.5,
            "bank": SimpleNamespace(incoming=bool(cmd["incoming"])),
            "abs_error_y": float(cmd["abs_error_y"]),
            "reach_miss": bool(cmd["reach_miss"]),
            "oracle_y": self.oracle_y,
            "escape_sign": self.escape_sign,
        }

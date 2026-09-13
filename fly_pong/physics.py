"""Pure Pong step. No pygame, no Gym."""

from __future__ import annotations

import math
from typing import Any

from fly_pong.constants import load_constants

State = dict[str, Any]


def clip(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def paddle_span(C: dict[str, Any] | None = None) -> float:
    C = C or load_constants()
    return float(C["height"] - C["paddleH"])


def clip_paddle(y: float, C: dict[str, Any] | None = None) -> float:
    C = C or load_constants()
    return clip(y, 0.0, paddle_span(C))


def cap_ball_speed(vx: float, vy: float, C: dict[str, Any] | None = None) -> tuple[float, float]:
    """Optional play-court cap. Locked evals omit ballSpeedMax and are unchanged."""
    C = C or load_constants()
    cap = C.get("ballSpeedMax")
    if cap is None:
        return float(vx), float(vy)
    cap = float(cap)
    spd = math.hypot(float(vx), float(vy))
    if spd <= cap or spd < 1e-9:
        return float(vx), float(vy)
    s = cap / spd
    return float(vx) * s, float(vy) * s


def dy_from_action(action: int, speed: float | None = None, C: dict[str, Any] | None = None) -> float:
    C = C or load_constants()
    if speed is None:
        speed = float(C["paddleSpeed"])
    action = int(action)
    if action == 1:
        return -speed
    if action == 2:
        return speed
    return 0.0


def mirror_right_state(raw: State, C: dict[str, Any] | None = None) -> State:
    """View the court from the right paddle, so approach_commit can drive it."""
    C = C or load_constants()
    w = float(C["width"])
    return {
        "ball_x": w - float(raw["ball_x"]),
        "ball_y": raw["ball_y"],
        "ball_vx": -float(raw["ball_vx"]),
        "ball_vy": raw["ball_vy"],
        "paddle_y": raw["opp_y"],
        "opp_y": raw["agent_y"],
        "width": C["width"],
        "height": C["height"],
        "paddle_h": C["paddleH"],
        "agent_score": raw.get("opp_score", 0),
        "opp_score": raw.get("agent_score", 0),
    }


def lag_opponent_dy(state: State, C: dict[str, Any] | None = None) -> float:
    """Chase the ball at oppLag × paddleSpeed. Training sparring partner."""
    C = C or load_constants()
    target = state["ball_y"] - C["paddleH"] / 2.0
    speed = float(C["paddleSpeed"]) * float(C["oppLag"])
    center = state["opp_y"] + C["paddleH"] / 2.0
    if center > target:
        return -speed
    return speed


def serve(state: State, direction: float, angle: float, C: dict[str, Any] | None = None) -> State:
    C = C or load_constants()
    speed = float(C["ballSpeed"])
    out = dict(state)
    out["ball_x"] = float(C["width"]) / 2.0
    out["ball_y"] = float(C["height"]) / 2.0
    out["ball_vx"] = float(direction) * speed * math.cos(angle)
    out["ball_vy"] = speed * math.sin(angle)
    out["serve_dir"] = float(direction)
    return out


def initial_state(
    C: dict[str, Any] | None = None,
    *,
    serve_dir: float = 1.0,
    angle: float = 0.0,
) -> State:
    C = C or load_constants()
    mid = paddle_span(C) / 2.0
    state: State = {
        "ball_x": float(C["width"]) / 2.0,
        "ball_y": float(C["height"]) / 2.0,
        "ball_vx": 0.0,
        "ball_vy": 0.0,
        "agent_y": mid,
        "opp_y": mid,
        "agent_score": 0,
        "opp_score": 0,
        "terminated": False,
        "serve_dir": float(serve_dir),
    }
    return serve(state, serve_dir, angle, C)


def _paddle_overlap(ball_x: float, ball_y: float, px: float, py: float, C: dict[str, Any]) -> bool:
    r = float(C["ballR"])
    pw = float(C["paddleW"])
    ph = float(C["paddleH"])
    return (px - r <= ball_x <= px + pw + r) and (py - r <= ball_y <= py + ph + r)


def step(
    state: State,
    agent_dy: float,
    opp_dy: float,
    C: dict[str, Any] | None = None,
    serve_angle: float = 0.0,
) -> tuple[State, float]:
    """Advance one frame. Reward is from the left (agent) paddle's point of view."""
    C = C or load_constants()
    if state["terminated"]:
        return dict(state), 0.0

    out = dict(state)
    out["agent_y"] = clip_paddle(state["agent_y"] + float(agent_dy), C)
    out["opp_y"] = clip_paddle(state["opp_y"] + float(opp_dy), C)

    ball_x = float(state["ball_x"]) + float(state["ball_vx"])
    ball_y = float(state["ball_y"]) + float(state["ball_vy"])
    ball_vx = float(state["ball_vx"])
    ball_vy = float(state["ball_vy"])

    r = float(C["ballR"])
    h = float(C["height"])
    w = float(C["width"])
    pw = float(C["paddleW"])
    ph = float(C["paddleH"])
    gain = float(C["bounceGain"])
    spin = float(C["spin"])

    if ball_y <= r:
        ball_y = r
        ball_vy = abs(ball_vy)
    elif ball_y >= h - r:
        ball_y = h - r
        ball_vy = -abs(ball_vy)

    if ball_vx < 0 and _paddle_overlap(ball_x, ball_y, float(C["agentX"]), out["agent_y"], C):
        ball_vx = abs(ball_vx) * gain
        offset = (ball_y - (out["agent_y"] + ph / 2.0)) / (ph / 2.0)
        ball_vy = ball_vy + offset * spin
        ball_x = float(C["agentX"]) + pw + r

    if ball_vx > 0 and _paddle_overlap(ball_x, ball_y, float(C["oppX"]), out["opp_y"], C):
        ball_vx = -abs(ball_vx) * gain
        offset = (ball_y - (out["opp_y"] + ph / 2.0)) / (ph / 2.0)
        ball_vy = ball_vy + offset * spin
        ball_x = float(C["oppX"]) - r

    ball_vx, ball_vy = cap_ball_speed(ball_vx, ball_vy, C)

    reward = 0.0
    if ball_x < -r:
        out["opp_score"] = int(state["opp_score"]) + 1
        reward = -1.0
        if out["opp_score"] >= int(C["maxScore"]):
            out["terminated"] = True
            out["ball_x"] = ball_x
            out["ball_y"] = clip(ball_y, r, h - r)
            out["ball_vx"] = ball_vx
            out["ball_vy"] = ball_vy
            return out, reward
        next_dir = -float(state["serve_dir"])
        return serve(out, next_dir, float(serve_angle), C), reward

    if ball_x > w + r:
        out["agent_score"] = int(state["agent_score"]) + 1
        reward = 1.0
        if out["agent_score"] >= int(C["maxScore"]):
            out["terminated"] = True
            out["ball_x"] = ball_x
            out["ball_y"] = clip(ball_y, r, h - r)
            out["ball_vx"] = ball_vx
            out["ball_vy"] = ball_vy
            return out, reward
        next_dir = -float(state["serve_dir"])
        return serve(out, next_dir, float(serve_angle), C), reward

    out["ball_x"] = ball_x
    out["ball_y"] = ball_y
    out["ball_vx"] = ball_vx
    out["ball_vy"] = ball_vy
    return out, reward


def paddle_contact(prev: State, now: State) -> bool:
    """True when vx flipped and no point was scored (paddle, not serve)."""
    if int(now["agent_score"]) != int(prev["agent_score"]):
        return False
    if int(now["opp_score"]) != int(prev["opp_score"]):
        return False
    pv = float(prev["ball_vx"])
    nv = float(now["ball_vx"])
    return pv * nv < 0.0 and abs(pv) > 1e-9

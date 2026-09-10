"""Closed-loop matches for the gated device vs lag or a clone."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from fly_pong.constants import load_constants
from fly_pong.device import FlyPongDevice
from fly_pong.physics import initial_state, lag_opponent_dy, step as physics_step


def _info_state(raw: dict[str, Any], C: dict[str, Any]) -> dict[str, Any]:
    return {
        "ball_x": raw["ball_x"],
        "ball_y": raw["ball_y"],
        "ball_vx": raw["ball_vx"],
        "ball_vy": raw["ball_vy"],
        "paddle_y": raw["agent_y"],
        "opp_y": raw["opp_y"],
        "width": C["width"],
        "height": C["height"],
        "paddle_h": C["paddleH"],
        "agent_score": raw["agent_score"],
        "opp_score": raw["opp_score"],
    }


def _mirror_right(raw: dict[str, Any], C: dict[str, Any]) -> dict[str, Any]:
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
        "agent_score": raw["opp_score"],
        "opp_score": raw["agent_score"],
    }


def play_match(
    device: FlyPongDevice,
    *,
    seed: int,
    opponent: Literal["lag", "clone"] = "lag",
    clone: FlyPongDevice | None = None,
    max_frames: int = 20_000,
    serve_angle: float | None = None,
) -> dict[str, Any]:
    C = load_constants()
    rng = np.random.default_rng(seed)
    ang = float(rng.uniform(-C["serveAngleMax"], C["serveAngleMax"]) if serve_angle is None else serve_angle)
    direction = float(rng.choice(np.array([-1.0, 1.0])))
    state = initial_state(C, serve_dir=direction, angle=ang)
    device.reset()
    if clone is not None:
        clone.reset()

    contacts = 0
    incoming = 0
    aim_on = 0
    aim_off_steer = 0
    open_hits = 0
    track = []
    g_move = []
    g_aim = []
    frames = 0
    ph = float(C["paddleH"])
    h = float(C["height"])

    while frames < max_frames and not state["terminated"]:
        info = _info_state(state, C)
        cmd = device.step_command(info)
        agent_dy = float(cmd["dy"])
        if cmd["aim_active"]:
            aim_on += 1
        elif abs(cmd["u_offset"]) > 0:
            pass
        if (not cmd["aim_active"]) and abs(agent_dy) > 0 and cmd["bank"].incoming is False:
            pass
        if cmd["bank"].incoming:
            incoming += 1
            if cmd["aim_active"] is False and abs(
                cmd["dy"] - float(np.clip(cmd["u_dy"], -1, 1)) * float(C["paddleSpeed"])
            ) > 1e-6:
                aim_off_steer += 1
        if opponent == "lag":
            opp_dy = lag_opponent_dy(state, C)
        else:
            if clone is None:
                raise ValueError("clone opponent needs a device")
            opp_cmd = clone.step_command(_mirror_right(state, C))
            opp_dy = float(opp_cmd["dy"])

        prev_vx = float(state["ball_vx"])
        prev_open = (h - (state["opp_y"] + ph)) - state["opp_y"]
        state, reward = physics_step(state, agent_dy, opp_dy, C, serve_angle=float(rng.uniform(-C["serveAngleMax"], C["serveAngleMax"])))
        if prev_vx < 0 and float(state["ball_vx"]) > 0:
            contacts += 1
            # outbound vy sign vs open side (positive vy is down)
            if prev_open > 0 and state["ball_vy"] > 0:
                open_hits += 1
            elif prev_open < 0 and state["ball_vy"] < 0:
                open_hits += 1
        center = float(state["agent_y"]) + ph / 2.0
        track.append(abs(center - float(state["ball_y"])) / h)
        g_move.append(cmd["g_move"])
        g_aim.append(cmd["g_aim"])
        frames += 1

    agent = int(state["agent_score"])
    opp = int(state["opp_score"])
    return {
        "seed": seed,
        "agent_score": agent,
        "opp_score": opp,
        "win": bool(agent > opp),
        "frames": frames,
        "hit_frame_cap": frames >= max_frames,
        "contacts": contacts,
        "incoming_frames": incoming,
        "aim_active_frames": aim_on,
        "contact_rate": (contacts / float(contacts + opp)) if (contacts + opp) else 0.0,
        "open_hit_rate": (open_hits / float(contacts)) if contacts else 0.0,
        "mean_track_err": float(np.mean(track)) if track else 1.0,
        "g_move": np.mean(np.stack(g_move), axis=0).tolist() if g_move else [],
        "g_aim": np.mean(np.stack(g_aim), axis=0).tolist() if g_aim else [],
    }

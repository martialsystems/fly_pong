"""Closed-loop matches for the gated device vs lag or a clone."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from fly_pong.constants import load_constants
from fly_pong.device import FlyPongDevice
from fly_pong.physics import initial_state, lag_opponent_dy, step as physics_step


def _y_flight(y: float, vy: float, t: float, h: float, r: float) -> tuple[float, bool]:
    """Integrate Y with wall reflections. Returns (y_at_t, hit_wall)."""
    wall = False
    remaining = float(t)
    pos = float(y)
    vel = float(vy)
    for _ in range(12):
        if remaining <= 1e-9 or abs(vel) < 1e-9:
            break
        if vel > 0:
            tw = (h - r - pos) / vel
        else:
            tw = (pos - r) / (-vel)
        if tw < 0:
            pos = pos + vel * remaining
            break
        if remaining <= tw:
            pos = pos + vel * remaining
            remaining = 0.0
            break
        remaining -= tw
        pos = h - r if vel > 0 else r
        vel = -vel
        wall = True
    return pos, wall


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
    window_on = 0
    abort_on = 0
    aim_off_steer = 0
    open_hits = 0
    contact_log: list[dict[str, Any]] = []
    pending: list[int] = []
    track = []
    g_move = []
    g_aim = []
    g_unused = []
    cx_trace = []
    mb_trace = []
    frames = 0
    ph = float(C["paddleH"])
    h = float(C["height"])

    while frames < max_frames and not state["terminated"]:
        info = _info_state(state, C)
        cmd = device.step_command(info)
        agent_dy = float(cmd["dy"])
        if cmd["aim_active"]:
            aim_on += 1
        if cmd.get("in_window"):
            window_on += 1
            if not cmd.get("commit"):
                abort_on += 1
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
        prev_vy = float(state["ball_vy"])
        prev_open = (h - (state["opp_y"] + ph)) - state["opp_y"]
        opp_center = float(state["opp_y"]) + ph / 2.0
        state, reward = physics_step(state, agent_dy, opp_dy, C, serve_angle=float(rng.uniform(-C["serveAngleMax"], C["serveAngleMax"])))
        if prev_vx < 0 and float(state["ball_vx"]) > 0:
            contacts += 1
            geo = (float(state["ball_y"]) - (float(state["agent_y"]) + ph / 2.0)) / (ph / 2.0)
            open_side = 1.0 if prev_open > 0 else (-1.0 if prev_open < 0 else 0.0)
            applied_u = float(cmd["u_offset"]) if cmd["aim_active"] else 0.0
            vx = float(state["ball_vx"])
            far = float(C["oppX"])
            t_land = (far - float(state["ball_x"])) / vx if vx > 1e-6 else 0.0
            land_y, wall = _y_flight(float(state["ball_y"]), float(state["ball_vy"]), t_land, h, float(C["ballR"]))
            spin = float(C["spin"])
            paddle_err = abs(float(cmd["paddle_center_px"]) - float(cmd["target_center_px"]))
            row = {
                "u_offset": applied_u,
                "geo_offset": float(geo),
                "open_side": open_side,
                "aim_active": bool(cmd["aim_active"]),
                "signed_open_cmd": int(applied_u * open_side > 0),
                "signed_open_geo": int(geo * open_side > 0) if open_side != 0 else 0,
                "opp_landing_dist": abs(land_y - opp_center),
                "incoming_vx": prev_vx,
                "incoming_vy": prev_vy,
                "contact_y": float(state["ball_y"]),
                "outbound_vy": float(state["ball_vy"]),
                "wall_before_landing": bool(wall),
                "gap_side_at_contact": open_side,
                "side_at_landing": 1.0 if land_y > opp_center else (-1.0 if land_y < opp_center else 0.0),
                "paddle_err_px": paddle_err,
                "inbound_dominates": bool(abs(prev_vy) > abs(applied_u * spin)),
                "point_won": None,
            }
            contact_log.append(row)
            pending.append(len(contact_log) - 1)
            if hasattr(device.encoder, "observe_contact"):
                device.encoder.observe_contact(float(row["signed_open_geo"]))
            if prev_open > 0 and state["ball_vy"] > 0:
                open_hits += 1
            elif prev_open < 0 and state["ball_vy"] < 0:
                open_hits += 1
        if reward != 0.0:
            won = reward > 0.0
            for idx in pending:
                contact_log[idx]["point_won"] = won
            pending = []
        center = float(state["agent_y"]) + ph / 2.0
        track.append(abs(center - float(state["ball_y"])) / h)
        g_move.append(cmd["g_move"])
        g_aim.append(cmd["g_aim"])
        if cmd.get("g_unused") is not None:
            g_unused.append(cmd["g_unused"])
        cx_trace.append(float(cmd.get("cx_heading") or 0.0))
        mb_trace.append(float(cmd.get("mb_value") or 0.5))
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
        "window_frames": window_on,
        "abort_frames": abort_on,
        "contact_rate": (contacts / float(contacts + opp)) if (contacts + opp) else 0.0,
        "open_hit_rate": (open_hits / float(contacts)) if contacts else 0.0,
        "mean_track_err": float(np.mean(track)) if track else 1.0,
        "g_move": np.mean(np.stack(g_move), axis=0).tolist() if g_move else [],
        "g_aim": np.mean(np.stack(g_aim), axis=0).tolist() if g_aim else [],
        "g_unused": np.mean(np.stack(g_unused), axis=0).tolist() if g_unused else [],
        "mean_cx_heading": float(np.mean(cx_trace)) if cx_trace else 0.0,
        "mean_mb_value": float(np.mean(mb_trace)) if mb_trace else 0.5,
        "mean_abs_u_offset": float(np.mean([abs(c["u_offset"]) for c in contact_log])) if contact_log else 0.0,
        "mean_abs_geo_offset": float(np.mean([abs(c["geo_offset"]) for c in contact_log])) if contact_log else 0.0,
        "signed_open": float(np.mean([c["signed_open_geo"] for c in contact_log])) if contact_log else 0.0,
        "signed_open_cmd": float(np.mean([c["signed_open_cmd"] for c in contact_log])) if contact_log else 0.0,
        "mean_opp_landing_dist": float(np.mean([c["opp_landing_dist"] for c in contact_log])) if contact_log else 0.0,
        "contact_n": len(contact_log),
        "leak_n": sum(1 for c in contact_log if c["signed_open_cmd"] == 1 and c["signed_open_geo"] == 0),
        "leak_rate": (
            sum(1 for c in contact_log if c["signed_open_cmd"] == 1 and c["signed_open_geo"] == 0)
            / float(len(contact_log))
            if contact_log
            else 0.0
        ),
        "contact_log": contact_log,
    }

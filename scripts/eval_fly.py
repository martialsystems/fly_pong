#!/usr/bin/env python3
"""Match win rate of the T4/T5 + centering loop vs the lag opponent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fly_pong.brain import MOTOR_DELAY_FRAMES, T4T5_TAU, delay_ms_assumed
from fly_pong.bridge import FlyBrainBridge
from fly_pong.env import FlyPongEnv
from fly_pong.physics import dy_from_action

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "logs" / "fly_gate.json"


def play_one(seed: int, max_frames: int) -> dict:
    env = FlyPongEnv(render_mode=None)
    brain = FlyBrainBridge(n_ommatidia=32)
    _, info = env.reset(seed=seed)
    brain.reset()
    errs: list[float] = []
    frames = 0
    try:
        while frames < max_frames:
            action, _neural = brain.brain_step(info["state"])
            dy = brain.motor.push(dy_from_action(int(action), C=env.unwrapped.C))
            delayed = 0
            if dy < -0.5:
                delayed = 1
            elif dy > 0.5:
                delayed = 2
            _obs, _reward, terminated, truncated, info = env.step(delayed)
            state = info["state"]
            ph = float(state["paddle_h"])
            center = float(state["paddle_y"]) + ph / 2.0
            errs.append(abs(center - float(state["ball_y"])) / float(state["height"]))
            frames += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    inner = env.unwrapped._state
    agent = int(inner["agent_score"])
    opp = int(inner["opp_score"])
    return {
        "seed": seed,
        "agent_score": agent,
        "opp_score": opp,
        "win": bool(agent > opp),
        "frames": frames,
        "hit_frame_cap": frames >= max_frames,
        "mean_track_err": float(np.mean(errs)) if errs else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    games = [play_one(args.seed + i, args.max_frames) for i in range(int(args.games))]
    wins = sum(1 for g in games if g["win"])
    agent_points = sum(g["agent_score"] for g in games)
    opp_points = sum(g["opp_score"] for g in games)
    points = agent_points + opp_points
    lock = {
        "n": int(args.games),
        "seed0": int(args.seed),
        "max_frames": int(args.max_frames),
        "opponent": "lag_chase",
        "controller": "t4t5_plus_retinotopic_centering",
        "motor_delay_frames": int(MOTOR_DELAY_FRAMES),
        "delay_ms_assumed": delay_ms_assumed(),
        "t4t5_tau": float(T4T5_TAU),
        "wins": wins,
        "losses": int(args.games) - wins,
        "win_rate": wins / float(args.games),
        "agent_points": agent_points,
        "opp_points": opp_points,
        "point_rate": (agent_points / float(points)) if points else 0.0,
        "mean_track_err": float(np.mean([g["mean_track_err"] for g in games])),
        "mean_frames": float(np.mean([g["frames"] for g in games])),
        "hit_frame_cap": sum(1 for g in games if g["hit_frame_cap"]),
        "games": games,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(
        f"wins={wins}/{args.games} win_rate={lock['win_rate']:.3f}  "
        f"points={agent_points}-{opp_points} point_rate={lock['point_rate']:.3f}  "
        f"mean_track_err={lock['mean_track_err']:.3f}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

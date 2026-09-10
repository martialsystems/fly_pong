#!/usr/bin/env python3
"""Contact %, point WR, game WR, gate histograms. Claim bans via pongforge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_KEYS, MOVE_KEYS
from fly_pong.play_device import play_match
from pongforge.gate import require_claims, require_readme_clean

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def _summarize(games: list[dict], controller: str) -> dict:
    n = len(games)
    wins = sum(1 for g in games if g["win"])
    agent_pts = sum(g["agent_score"] for g in games)
    opp_pts = sum(g["opp_score"] for g in games)
    pts = agent_pts + opp_pts
    contacts = sum(g["contacts"] for g in games)
    misses = opp_pts
    g_move = np.mean([g["g_move"] for g in games], axis=0).tolist() if games else []
    g_aim = np.mean([g["g_aim"] for g in games], axis=0).tolist() if games else []
    return {
        "controller": controller,
        "n": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": wins / float(n) if n else 0.0,
        "agent_points": agent_pts,
        "opp_points": opp_pts,
        "point_rate": (agent_pts / float(pts)) if pts else 0.0,
        "contact_rate": (contacts / float(contacts + misses)) if (contacts + misses) else 0.0,
        "mean_track_err": float(np.mean([g["mean_track_err"] for g in games])),
        "mean_open_hit_rate": float(np.mean([g["open_hit_rate"] for g in games])),
        "g_move": dict(zip(MOVE_KEYS, g_move)) if g_move else {},
        "g_aim": dict(zip(AIM_KEYS, g_aim)) if g_aim else {},
        "games": games,
        "claim": (
            f"{wins}/{n} matches vs lag, point_rate="
            f"{(agent_pts / float(pts)) if pts else 0:.3f}, contact_rate="
            f"{(contacts / float(contacts + misses)) if (contacts + misses) else 0:.3f}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=Path, default=ARTIFACTS / "device.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "device_move.json")
    parser.add_argument("--controller", default="device_move")
    parser.add_argument("--opponent", default="lag", choices=["lag", "clone"])
    args = parser.parse_args()

    device = FlyPongDevice()
    if args.weights.is_file():
        device.load(args.weights)
    games = [
        play_match(device, seed=args.seed + i, opponent=args.opponent)
        for i in range(int(args.games))
    ]
    lock = _summarize(games, args.controller)
    require_claims(
        lag_god=False,
        centering_is_god=False,
        selfplay_live_100_is_skill=False,
        hemibrain_search=False,
        neurokernel_play=False,
        thread_id="eval_device",
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    require_readme_clean(readme)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(lock["claim"])
    print("g_move", lock["g_move"])
    print("g_aim", lock["g_aim"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

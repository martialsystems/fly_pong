#!/usr/bin/env python3
"""Signed-open vs a Phase A returner. Hypothesis on disk first. Points are not a pass key."""

from __future__ import annotations

import json
from pathlib import Path

from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_KEYS, AIM_N
from fly_pong.play_device import play_match
from pongforge.gate import require_claims, require_readme_clean

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "logs" / "aim_sign_hypothesis.json"
OUT = ROOT / "logs" / "aim_sign.json"
WEIGHTS = ROOT / "artifacts" / "device.pt"


def _arm(name: str, agent: FlyPongDevice, returner: FlyPongDevice, n: int, seed0: int) -> dict:
    games = [
        play_match(agent, seed=seed0 + i, opponent="clone", clone=returner)
        for i in range(n)
    ]
    agent_pts = sum(g["agent_score"] for g in games)
    opp_pts = sum(g["opp_score"] for g in games)
    pts = agent_pts + opp_pts
    open_hit = sum(g["open_hit_rate"] * g["contacts"] for g in games)
    contacts = sum(g["contacts"] for g in games)
    signed = sum(g["signed_open"] * g["contacts"] for g in games)
    signed_cmd = sum(g["signed_open_cmd"] * g["contacts"] for g in games)
    land = sum(g["mean_opp_landing_dist"] * g["contacts"] for g in games)
    return {
        "arm": name,
        "wins": sum(1 for g in games if g["win"]),
        "n": n,
        "agent_points": agent_pts,
        "opp_points": opp_pts,
        "point_rate": (agent_pts / float(pts)) if pts else 0.0,
        "open_hit_rate": (open_hit / float(contacts)) if contacts else 0.0,
        "signed_open": (signed / float(contacts)) if contacts else 0.0,
        "signed_open_cmd": (signed_cmd / float(contacts)) if contacts else 0.0,
        "mean_opp_landing_dist": (land / float(contacts)) if contacts else 0.0,
        "mean_abs_u_offset": float(sum(g["mean_abs_u_offset"] for g in games) / n),
        "mean_abs_geo_offset": float(sum(g["mean_abs_geo_offset"] for g in games) / n),
        "g_aim": dict(zip(AIM_KEYS, games[0]["g_aim"])) if games and games[0]["g_aim"] else {},
        "games": games,
    }


def main() -> None:
    hyp = json.loads(HYP.read_text(encoding="utf-8"))
    if not hyp.get("written_before_run"):
        raise SystemExit("aim_sign_hypothesis.json must be written before this eval")
    n = int(hyp["n"])
    seed0 = int(hyp["seed0"])

    returner = FlyPongDevice(aim_n=0)
    if WEIGHTS.is_file():
        returner.load(WEIGHTS, aim=False)
    returner.aim_n = 0

    move_only = FlyPongDevice(aim_n=0)
    if WEIGHTS.is_file():
        move_only.load(WEIGHTS, aim=False)
    move_only.aim_n = 0

    move_aim = FlyPongDevice(aim_n=AIM_N)
    if WEIGHTS.is_file():
        move_aim.load(WEIGHTS, aim=True)

    move = _arm("move_only", move_only, returner, n, seed0)
    aim = _arm("move_aim", move_aim, returner, n, seed0)

    d_signed = float(aim["signed_open"] - move["signed_open"])
    d_open = float(aim["open_hit_rate"] - move["open_hit_rate"])
    d_pts = int(aim["agent_points"] - move["agent_points"])
    passed = bool(
        d_signed >= float(hyp["pass"]["signed_open_delta_min"])
        and d_open >= float(hyp["pass"]["open_hit_delta_min"])
    )
    smasher = bool(
        aim["mean_abs_geo_offset"] >= 0.7
        and 0.4 <= aim["signed_open"] <= 0.6
    )

    require_claims(thread_id="eval_aim_sign")
    require_readme_clean((ROOT / "README.md").read_text(encoding="utf-8"))

    out = {
        "hypothesis": "logs/aim_sign_hypothesis.json",
        "opponent": "frozen_phase_a",
        "move_only": {k: v for k, v in move.items() if k != "games"},
        "move_aim": {k: v for k, v in aim.items() if k != "games"},
        "delta_signed_open": d_signed,
        "delta_open_hit": d_open,
        "delta_agent_points": d_pts,
        "passed": passed,
        "random_corner_smasher": smasher,
        "selfplay": "unlocked" if passed else "locked",
        "log_line": hyp["log_line"],
        "games_move_only": move["games"],
        "games_move_aim": aim["games"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"passed={passed} smasher={smasher} d_signed={d_signed:.3f} d_open={d_open:.3f} "
        f"d_pts={d_pts} geo={aim['mean_abs_geo_offset']:.3f} land={aim['mean_opp_landing_dist']:.1f}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

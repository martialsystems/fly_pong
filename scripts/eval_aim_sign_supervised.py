#!/usr/bin/env python3
"""Eval supervised-sign aim vs frozen Phase A. Hypothesis on disk first."""

from __future__ import annotations

import json
from pathlib import Path

from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_KEYS, AIM_N
from fly_pong.play_device import play_match
from pongforge.gate import require_claims, require_readme_clean

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "logs" / "aim_sign_supervised_hypothesis.json"
OUT = ROOT / "logs" / "aim_sign_supervised.json"
MOVE_WEIGHTS = ROOT / "artifacts" / "device.pt"
AIM_WEIGHTS = ROOT / "artifacts" / "device_aim_sign.pt"


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
        "mean_abs_geo_offset": float(sum(g["mean_abs_geo_offset"] for g in games) / n),
        "mean_abs_u_offset": float(sum(g["mean_abs_u_offset"] for g in games) / n),
        "g_aim": dict(zip(AIM_KEYS, games[0]["g_aim"])) if games and games[0]["g_aim"] else {},
        "games": games,
    }


def main() -> None:
    hyp = json.loads(HYP.read_text(encoding="utf-8"))
    if not hyp.get("written_before_run"):
        raise SystemExit("aim_sign_supervised_hypothesis.json must be written before this eval")
    if not AIM_WEIGHTS.is_file():
        raise SystemExit(f"missing {AIM_WEIGHTS}")
    n = int(hyp["eval"]["n"])
    seed0 = int(hyp["eval"]["seed0"])
    base = float(hyp["eval"]["move_only_signed_open_baseline"])

    returner = FlyPongDevice(aim_n=0)
    if MOVE_WEIGHTS.is_file():
        returner.load(MOVE_WEIGHTS, aim=False)
    returner.aim_n = 0

    move_only = FlyPongDevice(aim_n=0)
    if MOVE_WEIGHTS.is_file():
        move_only.load(MOVE_WEIGHTS, aim=False)
    move_only.aim_n = 0

    move_aim = FlyPongDevice(aim_n=AIM_N)
    move_aim.load(AIM_WEIGHTS, aim=True)

    move = _arm("move_only", move_only, returner, n, seed0)
    aim = _arm("move_aim", move_aim, returner, n, seed0)

    d_geo = float(aim["signed_open"] - move["signed_open"])
    leak = float(aim["signed_open_cmd"] - aim["signed_open"])
    d_open = float(aim["open_hit_rate"] - move["open_hit_rate"])
    geo_ok = bool(aim["signed_open"] >= float(hyp["pass"]["signed_open_geo_min"]) and d_geo >= float(hyp["pass"]["signed_open_geo_delta_min"]))
    leak_ok = bool(leak <= float(hyp["pass"]["leak_max"]))
    passed = bool(geo_ok and leak_ok)
    bounce_limit = bool(aim["signed_open_cmd"] > move["signed_open_cmd"] + 0.05 and d_geo < 0.10)
    selfplay = bool(passed and d_open >= 0.08)

    require_claims(thread_id="eval_aim_sign_supervised")
    require_readme_clean((ROOT / "README.md").read_text(encoding="utf-8"))

    out = {
        "hypothesis": "logs/aim_sign_supervised_hypothesis.json",
        "opponent": "frozen_phase_a",
        "move_only": {k: v for k, v in move.items() if k != "games"},
        "move_aim": {k: v for k, v in aim.items() if k != "games"},
        "delta_signed_open_geo": d_geo,
        "delta_open_hit": d_open,
        "leak_cmd_minus_geo": leak,
        "passed": passed,
        "bounce_map_is_limit": bounce_limit,
        "aim_retired_as_head": bool((not passed) and bounce_limit),
        "selfplay": "unlocked" if selfplay else "locked",
        "log_line": hyp["log_line"],
        "baseline_signed_open": base,
        "games_move_only": move["games"],
        "games_move_aim": aim["games"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"passed={passed} bounce_limit={bounce_limit} geo={aim['signed_open']:.3f} "
        f"cmd={aim['signed_open_cmd']:.3f} leak={leak:.3f} d_open={d_open:.3f} |geo|={aim['mean_abs_geo_offset']:.3f}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen intercept Y in the 24-frame window. Hypothesis first. No C."""

from __future__ import annotations

import json
from pathlib import Path

from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_N
from fly_pong.play_device import play_match
from pongforge.gate import require_claims, require_readme_clean

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "logs" / "aim_setpoint_hypothesis.json"
OUT = ROOT / "logs" / "aim_setpoint.json"
MOVE_WEIGHTS = ROOT / "artifacts" / "device.pt"
AIM_WEIGHTS = ROOT / "artifacts" / "device_aim_sign.pt"


def _arm(name: str, agent: FlyPongDevice, returner: FlyPongDevice, n: int, seed0: int) -> dict:
    games = []
    for i in range(n):
        g = play_match(agent, seed=seed0 + i, opponent="clone", clone=returner)
        g = dict(g)
        g.pop("contact_log", None)
        games.append(g)
    agent_pts = sum(g["agent_score"] for g in games)
    opp_pts = sum(g["opp_score"] for g in games)
    pts = agent_pts + opp_pts
    contacts = sum(g["contacts"] for g in games)
    open_hit = sum(g["open_hit_rate"] * g["contacts"] for g in games)
    signed = sum(g["signed_open"] * g["contacts"] for g in games)
    leaks = sum(g["leak_n"] for g in games)
    return {
        "arm": name,
        "wins": sum(1 for g in games if g["win"]),
        "n": n,
        "agent_points": agent_pts,
        "opp_points": opp_pts,
        "point_rate": (agent_pts / float(pts)) if pts else 0.0,
        "open_hit_rate": (open_hit / float(contacts)) if contacts else 0.0,
        "signed_open": (signed / float(contacts)) if contacts else 0.0,
        "leak_rate": (leaks / float(contacts)) if contacts else 0.0,
        "mean_abs_geo_offset": float(sum(g["mean_abs_geo_offset"] for g in games) / n),
        "mean_opp_landing_dist": float(sum(g["mean_opp_landing_dist"] for g in games) / n),
        "games": games,
    }


def main() -> None:
    hyp = json.loads(HYP.read_text(encoding="utf-8"))
    if not hyp.get("written_before_run"):
        raise SystemExit("aim_setpoint_hypothesis.json must be written before this eval")
    n = int(hyp["eval"]["n"])
    seed0 = int(hyp["eval"]["seed0"])

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

    leak_ok = bool(aim["leak_rate"] <= float(hyp["pass"]["leak_max"]))
    geo_ok = bool(aim["signed_open"] >= float(hyp["pass"]["signed_open_geo_min"]))
    score_ok = bool(aim["wins"] >= int(hyp["pass"]["matches_min"]))
    passed = bool(leak_ok and geo_ok and score_ok)
    painter_not_player = bool(leak_ok and aim["wins"] <= 10)

    require_claims(thread_id="eval_aim_setpoint")
    require_readme_clean((ROOT / "README.md").read_text(encoding="utf-8"))

    out = {
        "hypothesis": "logs/aim_setpoint_hypothesis.json",
        "opponent": "frozen_phase_a",
        "move_only": {k: v for k, v in move.items() if k != "games"},
        "move_aim": {k: v for k, v in aim.items() if k != "games"},
        "leak_ok": leak_ok,
        "geo_ok": geo_ok,
        "score_ok": score_ok,
        "passed": passed,
        "painter_not_player": painter_not_player,
        "write_c_hypothesis": False,
        "selfplay": "locked",
        "log_line": hyp["log_line"],
        "games_move_only": move["games"],
        "games_move_aim": aim["games"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"passed={passed} leak={aim['leak_rate']:.3f} geo={aim['signed_open']:.3f} "
        f"wins={aim['wins']}/{n} painter_not_player={painter_not_player}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

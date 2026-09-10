#!/usr/bin/env python3
"""EXP 2: unused move-gate ablation. Hypothesis first. Lag 40/40 is not a title."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fly_pong.device import FlyPongDevice, UnusedMoveDevice
from fly_pong.features import MOVE_KEYS, UNUSED_MOVE_KEYS
from fly_pong.play_device import play_match
from pongforge.gate import require_claims, require_readme_clean

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "logs" / "unused_move_gates_hypothesis.json"
OUT = ROOT / "logs" / "unused_move_gates.json"
WEIGHTS = ROOT / "artifacts" / "device_unused_move.pt"
MOVE_WEIGHTS = ROOT / "artifacts" / "device.pt"


def _mass(g_move: list[float], g_unused: list[float]) -> dict[str, float]:
    vals = list(g_move) + list(g_unused)
    names = list(MOVE_KEYS) + list(UNUSED_MOVE_KEYS)
    if len(vals) != len(names):
        return {}
    return {k: float(v) for k, v in zip(names, vals)}


def _arm(name: str, agent, opponent: str, n: int, seed0: int, clone=None) -> dict:
    games = []
    for i in range(n):
        if opponent == "lag":
            g = play_match(agent, seed=seed0 + i, opponent="lag")
        else:
            g = play_match(agent, seed=seed0 + i, opponent="clone", clone=clone)
        g = dict(g)
        g.pop("contact_log", None)
        games.append(g)
    agent_pts = sum(g["agent_score"] for g in games)
    opp_pts = sum(g["opp_score"] for g in games)
    pts = agent_pts + opp_pts
    contacts = sum(g["contacts"] for g in games)
    g_move = np.mean([g["g_move"] for g in games], axis=0).tolist() if games else []
    g_unused = np.mean([g["g_unused"] for g in games], axis=0).tolist() if games and games[0]["g_unused"] else []
    mass = _mass(g_move, g_unused)
    return {
        "arm": name,
        "wins": sum(1 for g in games if g["win"]),
        "n": n,
        "agent_points": agent_pts,
        "opp_points": opp_pts,
        "point_rate": (agent_pts / float(pts)) if pts else 0.0,
        "contact_rate": float(sum(g["contact_rate"] for g in games) / n) if n else 0.0,
        "open_hit_rate": (sum(g["open_hit_rate"] * g["contacts"] for g in games) / float(contacts))
        if contacts
        else 0.0,
        "g_move_mass": mass,
        "error_y_mass": float(mass.get("error_y") or 0.0),
        "mean_cx_heading": float(sum(g.get("mean_cx_heading", 0.0) for g in games) / n) if n else 0.0,
        "mean_mb_value": float(sum(g.get("mean_mb_value", 0.5) for g in games) / n) if n else 0.5,
        "games": games,
    }


def main() -> None:
    hyp = json.loads(HYP.read_text(encoding="utf-8"))
    if not hyp.get("written_before_run"):
        raise SystemExit("unused_move_gates_hypothesis.json must be written before this eval")
    n = int(hyp["eval"]["n"])
    seed0 = int(hyp["eval"]["seed0"])
    floor = int(hyp["eval"]["move_only_floor_vs_phase_a"])
    lost_bar = float(hyp["pre_register"]["error_y_mass_min_to_call_lost"])

    agent = UnusedMoveDevice()
    if WEIGHTS.is_file():
        agent.load(WEIGHTS)

    lag = _arm("unused_vs_lag", agent, "lag", n, seed0)

    returner = FlyPongDevice(aim_n=0)
    if MOVE_WEIGHTS.is_file():
        returner.load(MOVE_WEIGHTS, aim=False)
    returner.aim_n = 0
    vs_a = _arm("unused_vs_phase_a", agent, "clone", n, seed0, clone=returner)

    error_y_lost = bool(lag["error_y_mass"] >= lost_bar)
    lag_is_title = False
    useful_vs_a = bool(vs_a["wins"] > floor)
    passed = bool((not error_y_lost) and useful_vs_a)

    require_claims(thread_id="eval_unused_move_gates")
    require_readme_clean((ROOT / "README.md").read_text(encoding="utf-8"))

    out = {
        "hypothesis": "logs/unused_move_gates_hypothesis.json",
        "vs_lag": {k: v for k, v in lag.items() if k != "games"},
        "vs_phase_a": {k: v for k, v in vs_a.items() if k != "games"},
        "error_y_lost": error_y_lost,
        "unused_vision_lost_to_error_y": error_y_lost,
        "lag_40_40_is_title": lag_is_title,
        "useful_vs_phase_a": useful_vs_a,
        "passed": passed,
        "write_c_hypothesis": False,
        "selfplay": "locked",
        "title": "unused-move-gates",
        "log_line": hyp["log_line"],
        "games_lag": lag["games"],
        "games_phase_a": vs_a["games"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"error_y={lag['error_y_mass']:.3f} lost={error_y_lost} "
        f"lag={lag['wins']}/{n} phase_a={vs_a['wins']}/{n} useful={useful_vs_a}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Move-only vs move+aim vs a Phase A returner. Hypothesis is written first."""

from __future__ import annotations

import json
from pathlib import Path

from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_KEYS, AIM_N
from fly_pong.play_device import play_match
from pongforge.gate import require_claims, require_readme_clean

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "logs" / "aim_hypothesis.json"
OUT = ROOT / "logs" / "aim_vs_returner.json"
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
    return {
        "arm": name,
        "wins": sum(1 for g in games if g["win"]),
        "n": n,
        "agent_points": agent_pts,
        "opp_points": opp_pts,
        "point_rate": (agent_pts / float(pts)) if pts else 0.0,
        "open_hit_rate": (open_hit / float(contacts)) if contacts else 0.0,
        "mean_abs_u_offset": float(sum(g["mean_abs_u_offset"] for g in games) / n),
        "mean_abs_geo_offset": float(sum(g["mean_abs_geo_offset"] for g in games) / n),
        "g_aim": dict(zip(AIM_KEYS, games[0]["g_aim"])) if games and games[0]["g_aim"] else {},
        "games": games,
    }


def main() -> None:
    hyp = json.loads(HYP.read_text(encoding="utf-8"))
    if not hyp.get("written_before_run"):
        raise SystemExit("aim_hypothesis.json must be written before this eval")
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
    # Clock channel is gone; load() drops incompatible aim weights.

    move = _arm("move_only", move_only, returner, n, seed0)
    aim = _arm("move_aim", move_aim, returner, n, seed0)

    d_open = float(aim["open_hit_rate"] - move["open_hit_rate"])
    d_pts = int(aim["agent_points"] - move["agent_points"])
    g = aim["g_aim"]
    gate_mass = max((float(g.get(k) or 0.0) for k in hyp["pass"]["gate_keys"]), default=0.0)
    clock_absent = "predicted_contact_t" not in AIM_KEYS
    passed = bool(
        (d_open >= float(hyp["pass"]["open_hit_delta_min"]) or d_pts >= int(hyp["pass"]["point_delta_min"]))
        and gate_mass >= float(hyp["pass"]["gate_mass_min"])
        and clock_absent
    )
    # Prose bar for unlocking C: open-hit clearly above chase, not just a point delta.
    selfplay_ok = bool(passed and aim["open_hit_rate"] >= 0.65)

    require_claims(thread_id="eval_aim_returner")
    require_readme_clean((ROOT / "README.md").read_text(encoding="utf-8"))

    out = {
        "hypothesis": "logs/aim_hypothesis.json",
        "opponent": "frozen_phase_a",
        "move_only": {k: v for k, v in move.items() if k != "games"},
        "move_aim": {k: v for k, v in aim.items() if k != "games"},
        "delta_open_hit": d_open,
        "delta_agent_points": d_pts,
        "gate_mass": gate_mass,
        "clock_absent": clock_absent,
        "passed": passed,
        "selfplay": "unlocked" if selfplay_ok else "locked",
        "wire": {
            "move_only_mean_abs_u_offset": move["mean_abs_u_offset"],
            "move_aim_mean_abs_u_offset": aim["mean_abs_u_offset"],
            "move_only_mean_abs_geo_offset": move["mean_abs_geo_offset"],
            "move_aim_mean_abs_geo_offset": aim["mean_abs_geo_offset"],
        },
        "log_line": hyp["log_line"],
        "games_move_only": move["games"],
        "games_move_aim": aim["games"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"passed={passed} d_open={d_open:.3f} d_pts={d_pts} gate_mass={gate_mass:.3f} "
        f"u_off={aim['mean_abs_u_offset']:.3f} geo={aim['mean_abs_geo_offset']:.3f}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

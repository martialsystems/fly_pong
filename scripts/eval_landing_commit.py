#!/usr/bin/env python3
"""Commit-if-reachable veto. Hypothesis first. No C. Not an aim title."""

from __future__ import annotations

import json
from pathlib import Path

from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_N
from fly_pong.play_device import play_match
from pongforge.gate import require_claims, require_readme_clean

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "logs" / "landing_commit_hypothesis.json"
OUT = ROOT / "logs" / "landing_commit.json"
MOVE_WEIGHTS = ROOT / "artifacts" / "device.pt"


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
    window = sum(g.get("window_frames", 0) for g in games)
    abort = sum(g.get("abort_frames", 0) for g in games)
    commit = sum(g.get("aim_active_frames", 0) for g in games)
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
        "window_frames": window,
        "commit_frames": commit,
        "abort_frames": abort,
        "abort_share": (abort / float(window)) if window else 0.0,
        "games": games,
    }


def main() -> None:
    hyp = json.loads(HYP.read_text(encoding="utf-8"))
    if not hyp.get("written_before_run"):
        raise SystemExit("landing_commit_hypothesis.json must be written before this eval")
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

    commit_agent = FlyPongDevice(aim_n=AIM_N)
    if MOVE_WEIGHTS.is_file():
        commit_agent.load(MOVE_WEIGHTS, aim=False)

    move = _arm("move_only", move_only, returner, n, seed0)
    commit = _arm("commit", commit_agent, returner, n, seed0)

    leak_ok = bool(commit["leak_rate"] <= float(hyp["pass"]["leak_max"]))
    geo_ok = bool(commit["signed_open"] >= float(hyp["pass"]["signed_open_geo_min"]))
    score_ok = bool(commit["wins"] >= int(hyp["pass"]["matches_min"]))
    passed = bool(leak_ok and geo_ok and score_ok)

    require_claims(thread_id="eval_landing_commit")
    require_readme_clean((ROOT / "README.md").read_text(encoding="utf-8"))

    out = {
        "hypothesis": "logs/landing_commit_hypothesis.json",
        "opponent": "frozen_phase_a",
        "move_only": {k: v for k, v in move.items() if k != "games"},
        "commit": {k: v for k, v in commit.items() if k != "games"},
        "leak_ok": leak_ok,
        "geo_ok": geo_ok,
        "score_ok": score_ok,
        "passed": passed,
        "write_c_hypothesis": False,
        "selfplay": "locked",
        "title": "commit-if-reachable",
        "log_line": hyp["log_line"],
        "games_move_only": move["games"],
        "games_commit": commit["games"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"passed={passed} leak={commit['leak_rate']:.3f} geo={commit['signed_open']:.3f} "
        f"wins={commit['wins']}/{n} abort_share={commit['abort_share']:.3f}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

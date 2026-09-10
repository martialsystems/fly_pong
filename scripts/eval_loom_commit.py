#!/usr/bin/env python3
"""EXP 1: loom commit/abort. Hypothesis first. No C. Not a title."""

from __future__ import annotations

import json
from pathlib import Path

from fly_pong.device import FlyPongDevice, LoomCommitDevice
from fly_pong.features import AIM_N
from fly_pong.play_device import play_match
from pongforge.gate import require_claims, require_readme_clean

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "logs" / "loom_commit_hypothesis.json"
OUT = ROOT / "logs" / "loom_commit.json"
MOVE_WEIGHTS = ROOT / "artifacts" / "device.pt"


def _arm(name: str, agent, returner: FlyPongDevice, n: int, seed0: int) -> dict:
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
    leaks = sum(g["leak_n"] for g in games)
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
        "signed_open": (sum(g["signed_open"] * g["contacts"] for g in games) / float(contacts))
        if contacts
        else 0.0,
        "leak_rate": (leaks / float(contacts)) if contacts else 0.0,
        "mean_cx_heading": float(sum(g.get("mean_cx_heading", 0.0) for g in games) / n) if n else 0.0,
        "mean_mb_value": float(sum(g.get("mean_mb_value", 0.5) for g in games) / n) if n else 0.5,
        "games": games,
    }


def main() -> None:
    hyp = json.loads(HYP.read_text(encoding="utf-8"))
    if not hyp.get("written_before_run"):
        raise SystemExit("loom_commit_hypothesis.json must be written before this eval")
    n = int(hyp["eval"]["n"])
    seed0 = int(hyp["eval"]["seed0"])
    floor = int(hyp["eval"]["move_only_floor"])

    returner = FlyPongDevice(aim_n=0)
    if MOVE_WEIGHTS.is_file():
        returner.load(MOVE_WEIGHTS, aim=False)
    returner.aim_n = 0

    move_only = FlyPongDevice(aim_n=0)
    if MOVE_WEIGHTS.is_file():
        move_only.load(MOVE_WEIGHTS, aim=False)
    move_only.aim_n = 0

    loom = LoomCommitDevice(aim_n=AIM_N)
    if MOVE_WEIGHTS.is_file():
        loom.load(MOVE_WEIGHTS, aim=False)

    move = _arm("move_only", move_only, returner, n, seed0)
    loom_arm = _arm("loom_commit", loom, returner, n, seed0)

    offset_off = True
    leak_ok = bool(loom_arm["leak_rate"] <= float(hyp["pass"]["leak_max"]))
    contact_ok = bool(loom_arm["contact_rate"] >= move["contact_rate"])
    leak_or_contact = bool(leak_ok or (offset_off and contact_ok))
    score_ok = bool(loom_arm["wins"] >= int(hyp["pass"]["matches_min"]))
    floor_ok = bool(loom_arm["wins"] >= floor)
    passed = bool(leak_or_contact and score_ok and floor_ok)

    require_claims(thread_id="eval_loom_commit")
    require_readme_clean((ROOT / "README.md").read_text(encoding="utf-8"))

    out = {
        "hypothesis": "logs/loom_commit_hypothesis.json",
        "opponent": "frozen_phase_a",
        "move_only": {k: v for k, v in move.items() if k != "games"},
        "loom_commit": {k: v for k, v in loom_arm.items() if k != "games"},
        "offset_off": offset_off,
        "leak_ok": leak_ok,
        "contact_ok": contact_ok,
        "score_ok": score_ok,
        "floor_ok": floor_ok,
        "passed": passed,
        "write_c_hypothesis": False,
        "selfplay": "locked",
        "title": "loom-commit-abort",
        "log_line": hyp["log_line"],
        "games_move_only": move["games"],
        "games_loom": loom_arm["games"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"passed={passed} leak={loom_arm['leak_rate']:.3f} contact={loom_arm['contact_rate']:.3f} "
        f"wins={loom_arm['wins']}/{n} move_only={move['wins']}/{n}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

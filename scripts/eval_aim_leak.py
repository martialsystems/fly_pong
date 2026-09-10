#!/usr/bin/env python3
"""Leak autopsy. No training. Same 40 seeds vs frozen Phase A."""

from __future__ import annotations

import json
from pathlib import Path

from fly_pong.constants import load_constants
from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_N
from fly_pong.play_device import play_match
from pongforge.gate import require_claims, require_readme_clean

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "logs" / "aim_leak_hypothesis.json"
OUT = ROOT / "logs" / "aim_leak.json"
MOVE_WEIGHTS = ROOT / "artifacts" / "device.pt"
AIM_WEIGHTS = ROOT / "artifacts" / "device_aim_sign.pt"


def _class(row: dict, paddle_speed: float) -> str:
    if row["paddle_err_px"] > paddle_speed:
        return "control_late"
    if row["wall_before_landing"]:
        return "physics_wall"
    if row["inbound_dominates"]:
        return "physics_inbound"
    return "other"


def main() -> None:
    hyp = json.loads(HYP.read_text(encoding="utf-8"))
    if not hyp.get("written_before_run"):
        raise SystemExit("aim_leak_hypothesis.json must be written before this eval")
    n = int(hyp["eval"]["n"])
    seed0 = int(hyp["eval"]["seed0"])
    speed = float(load_constants()["paddleSpeed"])

    returner = FlyPongDevice(aim_n=0)
    if MOVE_WEIGHTS.is_file():
        returner.load(MOVE_WEIGHTS, aim=False)
    returner.aim_n = 0
    agent = FlyPongDevice(aim_n=AIM_N)
    agent.load(AIM_WEIGHTS, aim=True)

    leaks = []
    landing = []
    contacts_n = 0
    for i in range(n):
        g = play_match(agent, seed=seed0 + i, opponent="clone", clone=returner)
        landing.append(g["mean_opp_landing_dist"])
        for row in g["contact_log"]:
            contacts_n += 1
            if row["signed_open_cmd"] == 1 and row["signed_open_geo"] == 0:
                rec = dict(row)
                rec["seed"] = seed0 + i
                rec["class"] = _class(row, speed)
                leaks.append(rec)

    n_leak = len(leaks)
    counts = {"control_late": 0, "physics_wall": 0, "physics_inbound": 0, "other": 0}
    for r in leaks:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    physics_n = counts["physics_wall"] + counts["physics_inbound"]
    physics_share = physics_n / float(n_leak) if n_leak else 0.0
    control_share = counts["control_late"] / float(n_leak) if n_leak else 0.0
    physics_cap = bool(n_leak and physics_share >= 0.5)
    control_cap = bool(n_leak and control_share > physics_share)

    require_claims(thread_id="eval_aim_leak")
    require_readme_clean((ROOT / "README.md").read_text(encoding="utf-8"))

    out = {
        "hypothesis": "logs/aim_leak_hypothesis.json",
        "n_games": n,
        "seed0": seed0,
        "contacts": contacts_n,
        "n_leak": n_leak,
        "leak_rate": n_leak / float(contacts_n) if contacts_n else 0.0,
        "mean_opp_landing_dist": float(sum(landing) / len(landing)) if landing else 0.0,
        "counts": counts,
        "physics_share": physics_share,
        "control_share": control_share,
        "physics_is_the_cap": physics_cap,
        "control_is_the_cap": control_cap,
        "write_c_hypothesis": False,
        "selfplay": "locked",
        "log_line": hyp["log_line"],
        "leaks": leaks,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"n_leak={n_leak}/{contacts_n} physics={physics_share:.3f} control={control_share:.3f} "
        f"physics_cap={physics_cap} control_cap={control_cap} land={out['mean_opp_landing_dist']:.1f}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Valence-flip approach commit. Hypothesis first. Not a fly result. Not placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from fly_pong.brain import decode_motor
from fly_pong.bridge import FlyBrainBridge
from fly_pong.commit import (
    COMMIT_WINDOW,
    ESCAPE_SIGN,
    ApproachCommitController,
    _paddle_center,
)
from fly_pong.constants import load_constants
from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_KEYS, AIM_N, MOVE_KEYS
from fly_pong.physics import dy_from_action
from fly_pong.play_device import play_match
from pongforge.gate import require_claims, require_readme_clean

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "logs" / "approach_commit_hypothesis.json"
DEFAULT_OUT = ROOT / "logs" / "approach_commit.json"
MOVE_WEIGHTS = ROOT / "artifacts" / "device.pt"


class FlyLoopAgent:
    """Motion-only or T4/T5 + centering. No commit gate."""

    def __init__(self, *, motion_only: bool = False, n_ommatidia: int = 32):
        self.bridge = FlyBrainBridge(n_ommatidia=n_ommatidia)
        self.motion_only = bool(motion_only)
        self.C = load_constants()
        self.aim_n = 0
        self.encoder = SimpleNamespace(observe_contact=lambda *_a, **_k: None)
        self.stats: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self.bridge.reset()
        self.stats = {
            "commit_frames": 0,
            "window_frames": 0,
            "reach_misses": 0,
            "total_frames": 0,
            "abs_error_at_commit": [],
        }

    def step_command(self, state: dict[str, Any]) -> dict[str, Any]:
        action, neural = self.bridge.brain_step(state)
        if self.motion_only:
            action = decode_motor(
                self.bridge.vel_gain * float(neural["steering"]),
                threshold=self.bridge.pos_threshold,
            )
        dy = float(dy_from_action(int(action), C=self.C))
        incoming = float(state["ball_vx"]) < 0.0
        y_paddle = _paddle_center(state, self.C)
        self.stats["total_frames"] += 1
        return {
            "dy": dy,
            "action": int(action),
            "u_dy": 0.0,
            "u_offset": 0.0,
            "u_offset_head": 0.0,
            "aim_active": False,
            "commit": False,
            "in_window": False,
            "reach": 0.0,
            "tau": 0.0,
            "target_center_px": y_paddle,
            "paddle_center_px": y_paddle,
            "g_move": np.zeros(len(MOVE_KEYS), dtype=np.float32),
            "g_aim": np.zeros(len(AIM_KEYS), dtype=np.float32),
            "cx_heading": 0.0,
            "mb_value": 0.5,
            "bank": SimpleNamespace(incoming=incoming),
        }


def _harvest(agent: Any) -> dict[str, Any]:
    st = getattr(agent, "stats", None) or {}
    total = int(st.get("total_frames") or 0)
    commit = int(st.get("commit_frames") or 0)
    window = int(st.get("window_frames") or 0)
    misses = int(st.get("reach_misses") or 0)
    errs = list(st.get("abs_error_at_commit") or [])
    return {
        "gate_commit_frames": commit,
        "gate_window_frames": window,
        "reach_misses": misses,
        "commit_frame_frac": (commit / float(total)) if total else 0.0,
        "mean_abs_error_y_at_commit": float(np.mean(errs)) if errs else 0.0,
        "total_frames_ctrl": total,
    }


def _arm(
    name: str,
    agent: Any,
    *,
    n: int,
    seed0: int,
    opponent: str,
    clone: FlyPongDevice | None = None,
) -> dict[str, Any]:
    games = []
    for i in range(n):
        if opponent == "lag":
            g = play_match(agent, seed=seed0 + i, opponent="lag")
        else:
            g = play_match(agent, seed=seed0 + i, opponent="clone", clone=clone)
        extra = _harvest(agent)
        g = dict(g)
        g.pop("contact_log", None)
        g.update(extra)
        games.append(g)
    agent_pts = sum(g["agent_score"] for g in games)
    opp_pts = sum(g["opp_score"] for g in games)
    pts = agent_pts + opp_pts
    contacts = sum(g["contacts"] for g in games)
    leaks = sum(g["leak_n"] for g in games)
    commit_f = sum(g["gate_commit_frames"] for g in games)
    window_f = sum(g["gate_window_frames"] for g in games)
    total_f = sum(g["total_frames_ctrl"] for g in games)
    misses = sum(g["reach_misses"] for g in games)
    err_w = [
        (g["mean_abs_error_y_at_commit"], g["gate_commit_frames"])
        for g in games
        if g["gate_commit_frames"]
    ]
    err_num = sum(m * w for m, w in err_w)
    err_den = sum(w for _m, w in err_w)
    return {
        "arm": name,
        "wins": sum(1 for g in games if g["win"]),
        "n": n,
        "agent_points": agent_pts,
        "opp_points": opp_pts,
        "point_rate": (agent_pts / float(pts)) if pts else 0.0,
        "contact_rate": float(sum(g["contact_rate"] for g in games) / n) if n else 0.0,
        "contact_rate_pooled": (contacts / float(contacts + opp_pts)) if (contacts + opp_pts) else 0.0,
        "leak_rate": (leaks / float(contacts)) if contacts else 0.0,
        "mean_track_err": float(np.mean([g["mean_track_err"] for g in games])) if games else 1.0,
        "open_hit_rate": (
            sum(g["open_hit_rate"] * g["contacts"] for g in games) / float(contacts)
            if contacts
            else 0.0
        ),
        "mean_abs_error_y_at_commit": (err_num / float(err_den)) if err_den else 0.0,
        "commit_frame_frac": (commit_f / float(total_f)) if total_f else 0.0,
        "commit_frames": commit_f,
        "window_frames": window_f,
        "reach_misses": misses,
        "reach_miss_rate": (misses / float(window_f)) if window_f else 0.0,
        "games": games,
    }


def _strip_games(arm: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in arm.items() if k != "games"}


def _lock_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"file": str(path.relative_to(ROOT))}
    if "controller" in data:
        out.update(
            {
                "n": data.get("n"),
                "wins": data.get("wins"),
                "agent_points": data.get("agent_points"),
                "opp_points": data.get("opp_points"),
                "mean_track_err": data.get("mean_track_err"),
                "controller": data.get("controller"),
                "opponent": data.get("opponent"),
            }
        )
        return out
    if "commit" in data:
        games = data.get("games_commit") or []
        contact = (
            float(sum(g["contact_rate"] for g in games) / len(games)) if games else None
        )
        out.update(
            {
                "wins": data["commit"]["wins"],
                "n": data["commit"]["n"],
                "agent_points": data["commit"]["agent_points"],
                "opp_points": data["commit"]["opp_points"],
                "leak_rate": data["commit"]["leak_rate"],
                "contact_rate": contact,
                "opponent": data.get("opponent"),
                "title": data.get("title"),
            }
        )
        return out
    if "loom_commit" in data:
        out.update(
            {
                "wins": data["loom_commit"]["wins"],
                "n": data["loom_commit"]["n"],
                "agent_points": data["loom_commit"]["agent_points"],
                "opp_points": data["loom_commit"]["opp_points"],
                "leak_rate": data["loom_commit"]["leak_rate"],
                "contact_rate": data["loom_commit"]["contact_rate"],
                "opponent": data.get("opponent"),
                "title": data.get("title"),
            }
        )
        return out
    return out


def _phase_a_returner() -> FlyPongDevice:
    returner = FlyPongDevice(aim_n=0)
    if MOVE_WEIGHTS.is_file():
        returner.load(MOVE_WEIGHTS, aim=False)
    returner.aim_n = 0
    return returner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--oracle_y",
        action="store_true",
        help="Cheat: true ball Y/vy. The bakeoff JSON always includes estimated and oracle arms.",
    )
    args = parser.parse_args()

    hyp = json.loads(HYP.read_text(encoding="utf-8"))
    if not hyp.get("written_before_run"):
        raise SystemExit("approach_commit_hypothesis.json must be written before this eval")
    n = int(args.games)
    seed0 = int(args.seed)
    if n != int(hyp["eval"]["n"]) or seed0 != int(hyp["eval"]["seed0"]):
        # Allow a shorter smoke run; the lock file uses the registered n/seed.
        pass

    landing = json.loads((ROOT / "logs" / "landing_commit.json").read_text(encoding="utf-8"))
    landing_games = landing.get("games_commit") or []
    landing_contact = (
        float(sum(g["contact_rate"] for g in landing_games) / len(landing_games))
        if landing_games
        else 0.0
    )

    motion = _arm("motion_only", FlyLoopAgent(motion_only=True), n=n, seed0=seed0, opponent="lag")
    centering = _arm("centering", FlyLoopAgent(motion_only=False), n=n, seed0=seed0, opponent="lag")
    approach_est = _arm(
        "approach_commit",
        ApproachCommitController(oracle_y=False, escape_sign=ESCAPE_SIGN),
        n=n,
        seed0=seed0,
        opponent="lag",
    )
    approach_oracle = _arm(
        "approach_commit_oracle_y",
        ApproachCommitController(oracle_y=True, escape_sign=ESCAPE_SIGN),
        n=n,
        seed0=seed0,
        opponent="lag",
    )

    returner = _phase_a_returner()
    approach_est_a = _arm(
        "approach_commit",
        ApproachCommitController(oracle_y=False, escape_sign=ESCAPE_SIGN),
        n=n,
        seed0=seed0,
        opponent="clone",
        clone=returner,
    )
    approach_oracle_a = _arm(
        "approach_commit_oracle_y",
        ApproachCommitController(oracle_y=True, escape_sign=ESCAPE_SIGN),
        n=n,
        seed0=seed0,
        opponent="clone",
        clone=returner,
    )

    contact_ok = bool(approach_est_a["contact_rate"] >= landing_contact)
    window_ok = bool(COMMIT_WINDOW == 24 and AIM_N == 24)
    passed = bool(contact_ok and window_ok)

    require_claims(thread_id="eval_approach_commit")
    require_readme_clean((ROOT / "README.md").read_text(encoding="utf-8"))

    out = {
        "hypothesis": "logs/approach_commit_hypothesis.json",
        "title": "valence-flip-approach-commit",
        "log_line": hyp["log_line"],
        "hypothesis_text": hyp["hypothesis"],
        "window_frames": int(COMMIT_WINDOW),
        "escape_sign_approach": -1,
        "escape_sign_flee": 1,
        "escape_sign_used": int(ESCAPE_SIGN),
        "oracle_y_flag": bool(args.oracle_y),
        "opponent_bakeoff": "lag_chase",
        "opponent_contact_bar": "frozen_phase_a",
        "landing_commit_contact_rate": landing_contact,
        "motion_only": _strip_games(motion),
        "centering": _strip_games(centering),
        "approach_commit": _strip_games(approach_est),
        "approach_commit_oracle_y": _strip_games(approach_oracle),
        "approach_commit_vs_phase_a": _strip_games(approach_est_a),
        "approach_commit_oracle_y_vs_phase_a": _strip_games(approach_oracle_a),
        "compare": {
            "fly_gate_motion_only": _lock_summary(ROOT / "logs" / "fly_gate_motion_only.json"),
            "fly_gate": _lock_summary(ROOT / "logs" / "fly_gate.json"),
            "landing_commit": _lock_summary(ROOT / "logs" / "landing_commit.json"),
            "loom_commit": _lock_summary(ROOT / "logs" / "loom_commit.json"),
        },
        "contact_ok": contact_ok,
        "window_ok": window_ok,
        "passed": passed,
        "open_hit_is_success_metric": False,
        "lag_40_40_is_title": False,
        "write_c_hypothesis": False,
        "selfplay": "locked",
        "public": "unchanged",
        "games_motion_only": motion["games"],
        "games_centering": centering["games"],
        "games_approach": approach_est["games"],
        "games_approach_oracle_y": approach_oracle["games"],
        "games_approach_vs_phase_a": approach_est_a["games"],
        "games_approach_oracle_y_vs_phase_a": approach_oracle_a["games"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"passed={passed} contact_phase_a={approach_est_a['contact_rate']:.3f} "
        f"landing={landing_contact:.3f} "
        f"lag approach={approach_est['wins']}/{n} centering={centering['wins']}/{n} "
        f"motion={motion['wins']}/{n} oracle={approach_oracle['wins']}/{n}"
    )
    print(
        f"oracle_y vs lag contact={approach_oracle['contact_rate']:.3f} "
        f"vs phase_a contact={approach_oracle_a['contact_rate']:.3f}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

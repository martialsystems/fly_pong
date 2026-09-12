from __future__ import annotations

import json
from pathlib import Path

from fly_pong.commit import (
    COMMIT_WINDOW,
    ESCAPE_SIGN,
    ApproachCommitController,
    approach_commit,
    reflected_crossing_y,
    should_commit,
)
from fly_pong.constants import load_constants
from fly_pong.features import AIM_N
from fly_pong.physics import initial_state

ROOT = Path(__file__).resolve().parents[1]


def _incoming_state(*, ball_y: float, paddle_y: float, vy: float = 2.0, dist: float = 40.0):
    C = load_constants()
    st = initial_state(C, serve_dir=-1.0, angle=0.0)
    st["ball_x"] = float(C["agentX"]) + float(C["paddleW"]) + float(C["ballR"]) + dist
    st["ball_vx"] = -5.0
    st["ball_y"] = ball_y
    st["ball_vy"] = vy
    st["agent_y"] = paddle_y
    return {
        **st,
        "paddle_y": paddle_y,
        "width": C["width"],
        "height": C["height"],
        "paddle_h": C["paddleH"],
    }


def test_window_stays_24():
    assert COMMIT_WINDOW == 24
    assert AIM_N == 24
    assert COMMIT_WINDOW == AIM_N


def test_escape_sign_constant_is_approach():
    assert ESCAPE_SIGN == -1


def test_approach_drives_toward_crossing_y():
    # t_to_line = 8, max travel 56 px. Paddle center 180, ball 220: reachable.
    st = _incoming_state(ball_y=220.0, paddle_y=150.0, vy=0.0, dist=40.0)
    cmd = approach_commit(st, oracle_y=True, escape_sign=-1)
    assert cmd["commit"] is True
    assert cmd["dy"] > 0.0
    flee = approach_commit(st, oracle_y=True, escape_sign=+1)
    assert flee["commit"] is True
    assert flee["dy"] < 0.0
    assert cmd["dy"] == -flee["dy"]


def test_flee_is_opposite_motor_sign():
    st = _incoming_state(ball_y=140.0, paddle_y=150.0, vy=0.0, dist=40.0)
    approach = approach_commit(st, oracle_y=True, escape_sign=-1)
    flee = approach_commit(st, oracle_y=True, escape_sign=+1)
    assert approach["commit"] and flee["commit"]
    assert approach["dy"] < 0.0
    assert flee["dy"] > 0.0


def test_no_commit_when_receding():
    st = _incoming_state(ball_y=180.0, paddle_y=150.0, vy=0.0, dist=40.0)
    st["ball_vx"] = 5.0
    cmd = approach_commit(st, oracle_y=True)
    assert cmd["incoming"] is False
    assert cmd["commit"] is False
    assert cmd["dy"] == 0.0 or abs(cmd["dy"]) <= 7.0


def test_abort_when_unreachable_is_reach_miss():
    st = _incoming_state(ball_y=float(load_constants()["ballR"]), paddle_y=300.0, vy=0.0, dist=20.0)
    cmd = approach_commit(st, oracle_y=True, escape_sign=-1)
    assert cmd["in_window"] is True
    assert cmd["commit"] is False
    assert cmd["reach_miss"] is True


def test_reflection_matches_one_bounce():
    C = load_constants()
    h = float(C["height"])
    r = float(C["ballR"])
    y = 350.0
    vy = 5.0
    t = 10.0
    got = reflected_crossing_y(y, vy, t, h, r)
    # Hit 354 at 0.8 frames, then 9.2 frames at vy=-5 → 354 - 46 = 308.
    assert abs(got - 308.0) < 1e-6
    flat = reflected_crossing_y(180.0, 2.0, 10.0, h, r)
    assert abs(flat - 200.0) < 1e-6


def test_oracle_uses_true_y_estimate_uses_passed_y():
    st = _incoming_state(ball_y=260.0, paddle_y=150.0, vy=0.0, dist=40.0)
    oracle = approach_commit(st, oracle_y=True)
    est = approach_commit(st, oracle_y=False, y_est=100.0, vy_est=0.0)
    assert abs(oracle["y_ball"] - 260.0) < 1e-6
    assert abs(est["y_ball"] - 100.0) < 1e-6
    assert oracle["y_target"] != est["y_target"]


def test_controller_steps_retina_and_remaps_motor():
    ctrl = ApproachCommitController(oracle_y=True, escape_sign=-1)
    ctrl.reset()
    st = _incoming_state(ball_y=220.0, paddle_y=150.0, vy=0.0, dist=40.0)
    cmd = ctrl.step_command(st)
    assert cmd["u_offset"] == 0.0
    assert cmd["commit"] is True
    assert cmd["dy"] > 0.0
    assert ctrl.stats["commit_frames"] == 1
    assert should_commit(
        incoming=True,
        tau=cmd["tau"],
        y_pred=cmd["target_center_px"],
        paddle_center=cmd["paddle_center_px"],
        paddle_speed=float(load_constants()["paddleSpeed"]),
        window=24.0,
    ) is True


def test_hypothesis_written_before_run():
    hyp = json.loads((ROOT / "logs" / "approach_commit_hypothesis.json").read_text(encoding="utf-8"))
    assert hyp["written_before_run"] is True
    assert hyp["title"] == "valence-flip-approach-commit"
    assert hyp["eval"]["n"] == 40
    assert hyp["eval"]["seed0"] == 0
    assert hyp["rule"]["tau_window"] == 24
    assert hyp["write_c_hypothesis"] is False
    assert hyp["selfplay"] == "locked"
    assert hyp["lag_40_40_is_title"] is False
    assert hyp["pass"]["open_hit_is_success_metric"] is False
    assert "occupy-the-crossing-Y" in hyp["hypothesis"]
    assert hyp["widen_window"] is False
    assert hyp["public"] == "unchanged"


def test_approach_commit_lock_and_docs():
    hyp = json.loads((ROOT / "logs" / "approach_commit_hypothesis.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "logs" / "approach_commit.json").read_text(encoding="utf-8"))
    assert res["passed"] is True
    assert res["write_c_hypothesis"] is False
    assert res["selfplay"] == "locked"
    assert res["lag_40_40_is_title"] is False
    assert res["open_hit_is_success_metric"] is False
    assert res["window_frames"] == 24
    assert res["escape_sign_used"] == -1
    assert res["public"] == "unchanged"
    assert res["approach_commit_vs_phase_a"]["contact_rate"] >= res["landing_commit_contact_rate"]
    assert res["approach_commit_vs_phase_a"]["leak_rate"] == 0.0
    assert res["approach_commit"]["leak_rate"] == 0.0
    assert res["approach_commit_oracle_y"]["n"] == 40
    assert res["motion_only"]["wins"] == 0
    assert "approach_commit" in res
    assert "approach_commit_oracle_y" in res
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert hyp["log_line"] in agents
    assert "valence flip" in text.lower()
    assert "not a fly result" in text
    assert "not a placement result" in text
    assert "ESCAPE_SIGN" in text
    assert "logs/approach_commit.json" in text
    assert "0.974" in text
    assert "eval_approach.py" in text
    assert "--oracle_y" in text
    assert "—" not in text
    assert "What it is not" not in text
    from pongforge.gate import scan_text_flags

    assert not any(scan_text_flags(text).values())

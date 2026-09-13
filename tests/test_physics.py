from __future__ import annotations

from fly_pong.constants import load_constants
from fly_pong.physics import cap_ball_speed, clip_paddle, dy_from_action, initial_state, paddle_contact, step


def test_paddle_clip():
    C = load_constants()
    assert clip_paddle(-10, C) == 0
    assert clip_paddle(10_000, C) == C["height"] - C["paddleH"]


def test_dy_from_action():
    C = load_constants()
    assert dy_from_action(0, C=C) == 0
    assert dy_from_action(1, C=C) == -C["paddleSpeed"]
    assert dy_from_action(2, C=C) == C["paddleSpeed"]


def test_agent_paddle_contact_flips_vx():
    C = load_constants()
    state = initial_state(C, serve_dir=-1.0, angle=0.0)
    state["ball_vx"] = -5.0
    state["ball_vy"] = 0.0
    state["ball_y"] = state["agent_y"] + float(C["paddleH"]) / 2.0
    state["ball_x"] = float(C["agentX"]) + float(C["paddleW"]) + float(C["ballR"]) + 2.0
    nxt, reward = step(state, 0.0, 0.0, C, serve_angle=0.0)
    assert reward == 0.0
    assert nxt["ball_vx"] > 0.0
    assert paddle_contact(state, nxt) is True


def test_score_is_not_a_paddle_contact():
    prev = {
        "agent_score": 0,
        "opp_score": 0,
        "ball_vx": -5.0,
    }
    now = {
        "agent_score": 0,
        "opp_score": 1,
        "ball_vx": 5.0,
    }
    assert paddle_contact(prev, now) is False


def test_default_constants_do_not_cap_speed():
    C = load_constants()
    assert C["bounceGain"] == 1.02
    assert "ballSpeedMax" not in C
    vx, vy = cap_ball_speed(20.0, 15.0, C)
    assert vx == 20.0 and vy == 15.0


def test_play_cap_scales_vector():
    C = dict(load_constants())
    C["ballSpeedMax"] = 10.0
    vx, vy = cap_ball_speed(6.0, 8.0, C)
    assert abs((vx * vx + vy * vy) ** 0.5 - 10.0) < 1e-6
    assert abs(vx / vy - 6.0 / 8.0) < 1e-6


def test_wall_bounce_flips_vy():
    C = load_constants()
    state = initial_state(C, serve_dir=1.0, angle=0.0)
    state["ball_y"] = 2.0
    state["ball_vy"] = -8.0
    state["ball_vx"] = 0.0
    nxt, reward = step(state, 0.0, 0.0, C, serve_angle=0.0)
    assert reward == 0.0
    assert nxt["ball_vy"] > 0


def test_left_miss_scores_opponent():
    C = load_constants()
    state = initial_state(C, serve_dir=1.0, angle=0.0)
    state["ball_x"] = -20
    state["ball_vx"] = -8
    nxt, reward = step(state, 0.0, 0.0, C, serve_angle=0.0)
    assert reward == -1.0
    assert nxt["opp_score"] == 1
    assert nxt["agent_score"] == 0
    assert nxt["serve_dir"] == -1.0


def test_right_miss_scores_agent():
    C = load_constants()
    state = initial_state(C, serve_dir=1.0, angle=0.0)
    state["ball_x"] = C["width"] + 20
    state["ball_vx"] = 8
    nxt, reward = step(state, 0.0, 0.0, C, serve_angle=0.0)
    assert reward == 1.0
    assert nxt["agent_score"] == 1


def test_match_point_terminates():
    C = load_constants()
    state = initial_state(C, serve_dir=1.0, angle=0.0)
    state["agent_score"] = C["maxScore"] - 1
    state["ball_x"] = C["width"] + 20
    state["ball_vx"] = 8
    nxt, reward = step(state, 0.0, 0.0, C, serve_angle=0.0)
    assert nxt["terminated"] is True
    assert nxt["agent_score"] == C["maxScore"]
    nxt2, r2 = step(nxt, 0.0, 0.0, C)
    assert r2 == 0.0
    assert nxt2["agent_score"] == C["maxScore"]

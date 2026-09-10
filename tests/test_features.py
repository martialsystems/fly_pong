from __future__ import annotations

from fly_pong.constants import load_constants
from fly_pong.features import AIM_KEYS, MOVE_KEYS, FeatureEncoder, desired_offset, frames_to_agent_paddle
from fly_pong.physics import initial_state


def test_move_and_aim_names():
    enc = FeatureEncoder()
    C = load_constants()
    st = initial_state(C, serve_dir=-1.0, angle=0.2)
    st_info = {
        **st,
        "paddle_y": st["agent_y"],
        "width": C["width"],
        "height": C["height"],
        "paddle_h": C["paddleH"],
    }
    bank = enc.encode(st_info)
    assert bank.names_move == MOVE_KEYS
    assert bank.names_aim == AIM_KEYS
    assert bank.move.shape == (len(MOVE_KEYS),)
    assert bank.aim.shape == (len(AIM_KEYS),)


def test_time_to_own_goal_incoming_vs_receding():
    C = load_constants()
    incoming = initial_state(C, serve_dir=-1.0, angle=0.0)
    receding = initial_state(C, serve_dir=1.0, angle=0.0)
    t_in = frames_to_agent_paddle(incoming, C)
    t_out = frames_to_agent_paddle(receding, C)
    assert t_in < t_out
    assert t_out >= 120.0 - 1e-6


def test_desired_offset_sign_matches_open_side():
    C = load_constants()
    st = initial_state(C)
    st["opp_y"] = 0.0
    assert desired_offset(st, C) > 0
    st["opp_y"] = float(C["height"] - C["paddleH"])
    assert desired_offset(st, C) < 0

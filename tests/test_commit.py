from __future__ import annotations

from fly_pong.commit import reach_frames, should_commit
from fly_pong.constants import load_constants
from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_N
from fly_pong.physics import initial_state


def test_reach_is_distance_over_speed():
    assert reach_frames(100.0, 30.0, 7.0) == 10.0
    assert reach_frames(30.0, 30.0, 7.0) == 0.0


def test_abort_when_unreachable_in_window():
    assert should_commit(
        incoming=True,
        tau=4.0,
        y_pred=0.0,
        paddle_center=180.0,
        paddle_speed=7.0,
        window=24.0,
    ) is False


def test_commit_when_reachable_in_window():
    assert should_commit(
        incoming=True,
        tau=10.0,
        y_pred=100.0,
        paddle_center=70.0,
        paddle_speed=7.0,
        window=24.0,
    ) is True


def test_no_commit_outside_window_even_if_aligned():
    assert should_commit(
        incoming=True,
        tau=80.0,
        y_pred=180.0,
        paddle_center=180.0,
        paddle_speed=7.0,
        window=24.0,
    ) is False


def test_no_commit_when_receding():
    assert should_commit(
        incoming=False,
        tau=4.0,
        y_pred=180.0,
        paddle_center=180.0,
        paddle_speed=7.0,
        window=24.0,
    ) is False


def _info(state, C):
    return {
        "ball_x": state["ball_x"],
        "ball_y": state["ball_y"],
        "ball_vx": state["ball_vx"],
        "ball_vy": state["ball_vy"],
        "paddle_y": state["agent_y"],
        "opp_y": state["opp_y"],
        "width": C["width"],
        "height": C["height"],
        "paddle_h": C["paddleH"],
    }


def test_device_does_not_lunge_to_offset():
    C = load_constants()
    device = FlyPongDevice(aim_n=AIM_N)
    st = initial_state(C, serve_dir=-1.0, angle=0.0)
    st["ball_x"] = float(C["agentX"]) + float(C["paddleW"]) + float(C["ballR"]) + 20.0
    st["ball_vx"] = -5.0
    cmd = device.step_command(_info(st, C))
    assert cmd["commit"] is True
    assert cmd["u_offset"] == 0.0
    assert abs(cmd["target_center_px"] - cmd["bank"].predicted_contact_y_px) < 1e-6


def test_device_aborts_to_error_y_when_unreachable():
    C = load_constants()
    device = FlyPongDevice(aim_n=AIM_N)
    st = initial_state(C, serve_dir=-1.0, angle=0.0)
    st["ball_x"] = float(C["agentX"]) + float(C["paddleW"]) + float(C["ballR"]) + 20.0
    st["ball_vx"] = -5.0
    st["ball_y"] = float(C["ballR"])
    st["agent_y"] = float(C["height"] - C["paddleH"])
    cmd = device.step_command(_info(st, C))
    assert cmd["in_window"] is True
    assert cmd["reach"] > cmd["tau"]
    assert cmd["commit"] is False
    assert cmd["aim_active"] is False
    assert cmd["u_offset"] == 0.0

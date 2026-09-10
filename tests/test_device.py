from __future__ import annotations

from fly_pong.constants import load_constants
from fly_pong.device import FlyPongDevice
from fly_pong.features import AIM_N
from fly_pong.physics import initial_state


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


def test_aim_idle_when_far_or_receding():
    C = load_constants()
    device = FlyPongDevice(aim_n=AIM_N)
    receding = initial_state(C, serve_dir=1.0, angle=0.0)
    cmd = device.step_command(_info(receding, C))
    assert cmd["aim_active"] is False
    far = initial_state(C, serve_dir=-1.0, angle=0.0)
    far["ball_x"] = float(C["width"]) * 0.85
    far["ball_vx"] = -2.0
    device.reset()
    cmd2 = device.step_command(_info(far, C))
    assert cmd2["bank"].frames_to_paddle > AIM_N
    assert cmd2["aim_active"] is False


def test_aim_active_near_contact():
    C = load_constants()
    device = FlyPongDevice(aim_n=AIM_N)
    st = initial_state(C, serve_dir=-1.0, angle=0.0)
    st["ball_x"] = float(C["agentX"]) + float(C["paddleW"]) + float(C["ballR"]) + 20.0
    st["ball_vx"] = -5.0
    cmd = device.step_command(_info(st, C))
    assert cmd["bank"].incoming is True
    assert cmd["bank"].frames_to_paddle <= AIM_N
    assert cmd["aim_active"] is True

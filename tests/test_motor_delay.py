from __future__ import annotations

import numpy as np

from fly_pong.brain import (
    MOTOR_DELAY_FRAMES,
    T4T5_TAU,
    MotorDelay,
    T4T5MotionCircuit,
    decode_motor,
    delay_ms_assumed,
)
from fly_pong.bridge import FlyBrainBridge
from fly_pong.constants import load_constants
from fly_pong.features import AIM_N
from fly_pong.sensors import CompoundEye


def test_tau_is_mi1_tm3_sized():
    assert T4T5_TAU == 1.5
    circuit = T4T5MotionCircuit(n=8)
    assert abs(circuit.tau - 1.5) < 1e-12


def test_motor_delay_is_two_frames_next_to_fps():
    C = load_constants()
    assert C["fps"] == 60
    assert MOTOR_DELAY_FRAMES == 2
    assert MOTOR_DELAY_FRAMES <= 3
    assert delay_ms_assumed() == MOTOR_DELAY_FRAMES * 1000.0 / float(C["fps"])
    delay = MotorDelay()
    assert delay.n == 2
    assert abs(delay.delay_ms_assumed - 1000.0 / 30.0) < 1e-12


def test_fifo_reads_oldest_sample():
    delay = MotorDelay(frames=2)
    assert delay.push(7.0) == 0.0
    assert delay.push(7.0) == 0.0
    assert delay.push(7.0) == 7.0
    assert delay.push(-7.0) == 7.0
    assert delay.push(0.0) == 7.0
    assert delay.push(0.0) == -7.0


def test_centering_instant_in_circuit_late_on_paddle():
    brain = FlyBrainBridge(n_ommatidia=32)
    high = {
        "height": 360.0,
        "ball_y": 80.0,
        "ball_x": 240.0,
        "paddle_y": 240.0,
        "paddle_h": 60.0,
        "ball_vx": -3.0,
    }
    action, neural = brain.brain_step(high)
    assert neural["pos_err"] < 0
    assert action == 1
    assert brain.motor.push(7.0) == 0.0


def test_motion_only_still_blind_to_still_blob():
    circuit = T4T5MotionCircuit(n=32)
    eye = CompoundEye(n_ommatidia=32)
    field = eye.encode({"height": 360.0, "ball_y": 180.0})
    last = None
    for _ in range(8):
        last = circuit.step(field)
    assert last is not None
    assert abs(last["steering"]) < 1e-6
    assert decode_motor(last["steering"]) == 0


def test_commit_window_not_widened():
    assert AIM_N == 24


def test_fps_unchanged():
    assert load_constants()["fps"] == 60

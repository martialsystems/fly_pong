from __future__ import annotations

import numpy as np

from fly_pong.brain import T4T5MotionCircuit, decode_motor
from fly_pong.bridge import FlyBrainBridge
from fly_pong.fbl_adapter import FlyBrainLabAdapter
from fly_pong.sensors import CompoundEye


def _moving_blob(n: int, y0: float, y1: float, frames: int) -> list[np.ndarray]:
    eye = CompoundEye(n_ommatidia=n)
    out = []
    for t in range(frames):
        y = y0 + (y1 - y0) * t / max(frames - 1, 1)
        out.append(eye.encode({"height": 360.0, "ball_y": y * 360.0}))
    return out


def test_downward_blob_steers_down():
    circuit = T4T5MotionCircuit(n=32)
    fields = _moving_blob(32, 0.2, 0.8, 12)
    last = None
    for field in fields:
        last = circuit.step(field)
    assert last is not None
    assert last["steering"] > 0
    assert decode_motor(last["steering"]) == 2


def test_upward_blob_steers_up():
    circuit = T4T5MotionCircuit(n=32)
    fields = _moving_blob(32, 0.8, 0.2, 12)
    last = None
    for field in fields:
        last = circuit.step(field)
    assert last is not None
    assert last["steering"] < 0
    assert decode_motor(last["steering"]) == 1


def test_bridge_reset_and_step():
    brain = FlyBrainBridge(n_ommatidia=16)
    brain.reset()
    action, neural = brain.brain_step(
        {"height": 360.0, "ball_y": 180.0, "ball_x": 240.0}
    )
    assert action in (0, 1, 2)
    assert "T4c" in neural and "stimulus" in neural


def test_fbl_adapter_is_stub():
    adapter = FlyBrainLabAdapter()
    try:
        adapter.inject_stimulus([0.0])
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass
    try:
        adapter.read_motor()
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass

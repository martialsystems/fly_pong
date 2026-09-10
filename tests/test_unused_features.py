from __future__ import annotations

import numpy as np

from fly_pong.constants import load_constants
from fly_pong.features import (
    UNUSED_MOVE_KEYS,
    FeatureEncoder,
    cx_heading,
    hs_close,
    lc4_loom_center,
    lc10_track,
    lc11_dark,
    lc18_fine,
    lpi_opp,
    lplc_loom,
)
from fly_pong.physics import initial_state


def _info(st, C):
    return {
        **st,
        "paddle_y": st["agent_y"],
        "width": C["width"],
        "height": C["height"],
        "paddle_h": C["paddleH"],
    }


def test_unused_keys_and_bank_shape():
    enc = FeatureEncoder()
    C = load_constants()
    bank = enc.encode(_info(initial_state(C, serve_dir=-1.0, angle=0.2), C))
    assert bank.names_unused == UNUSED_MOVE_KEYS
    assert bank.unused.shape == (len(UNUSED_MOVE_KEYS),)
    assert bank.cx_heading in (-1.0, 0.0, 1.0)
    assert 0.0 <= bank.mb_value <= 1.0


def test_hs_close_incoming_not_receding():
    C = load_constants()
    incoming = initial_state(C, serve_dir=-1.0, angle=0.0)
    receding = initial_state(C, serve_dir=1.0, angle=0.0)
    assert hs_close(incoming, C) > 0.0
    assert hs_close(receding, C) == 0.0


def test_lpi_opp_follows_preferred_direction():
    down = {"T4c": 0.0, "T4d": 0.1, "T5c": 0.0, "T5d": 0.1}
    up = {"T4c": 0.1, "T4d": 0.0, "T5c": 0.1, "T5d": 0.0}
    assert lpi_opp(down, ball_vy=1.0) > lpi_opp(down, ball_vy=-1.0)
    assert lpi_opp(up, ball_vy=-1.0) > lpi_opp(up, ball_vy=1.0)


def test_lc10_needs_motion():
    C = load_constants()
    luma = np.ones(32, dtype=np.float32)
    still = {"ball_vx": 0.0, "ball_vy": 0.0}
    moving = {"ball_vx": -5.0, "ball_vy": 4.0}
    assert lc10_track(luma, still, C) == 0.0
    assert lc10_track(luma, moving, C) > 0.0


def test_lc11_dark_on_flat_vs_peak():
    flat = np.full(32, 0.5, dtype=np.float32)
    peak = np.zeros(32, dtype=np.float32)
    peak[16] = 1.0
    assert lc11_dark(peak) > lc11_dark(flat)


def test_lc18_fine_needs_local_change():
    luma = np.zeros(32, dtype=np.float32)
    luma[16] = 1.0
    prev = np.zeros(32, dtype=np.float32)
    assert lc18_fine(luma, prev, ball_y=180.0, height=360.0) > 0.0
    assert lc18_fine(luma, luma, ball_y=180.0, height=360.0) == 0.0


def test_lplc_and_lc4_own_half():
    C = load_constants()
    st = initial_state(C, serve_dir=-1.0, angle=0.0)
    st["ball_x"] = float(C["width"]) * 0.25
    st["ball_vx"] = -4.0
    assert 0.0 < lplc_loom(st, C) < 1.0
    assert lc4_loom_center(st, C) == lplc_loom(st, C)
    st["ball_x"] = float(C["width"]) * 0.8
    assert lplc_loom(st, C) > 0.0
    assert lc4_loom_center(st, C) == 0.0
    st["ball_vx"] = 4.0
    assert lplc_loom(st, C) == 1.0
    assert lc4_loom_center(st, C) == 0.0


def test_cx_heading_is_ternary():
    C = load_constants()
    st = initial_state(C)
    st["opp_y"] = 0.0
    assert cx_heading(st, C) == 1.0
    st["opp_y"] = float(C["height"] - C["paddleH"])
    assert cx_heading(st, C) == -1.0


def test_mb_value_is_running_mean_of_contacts():
    enc = FeatureEncoder()
    assert enc.mb_value() == 0.5
    enc.observe_contact(1.0)
    enc.observe_contact(0.0)
    assert abs(enc.mb_value() - 0.5) < 1e-6
    enc.observe_contact(1.0)
    assert enc.mb_value() > 0.5


def test_cx_mb_not_in_move_bank():
    from fly_pong.features import MOVE_KEYS

    assert "CX_heading" not in MOVE_KEYS
    assert "MB_value" not in MOVE_KEYS
    assert "HS_close" not in MOVE_KEYS

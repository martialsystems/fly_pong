from __future__ import annotations

import pytest

from pongforge.gate import LawBlockedError, require_claims, require_phase, scan_text_flags


def test_phase_order_refuses_aim_before_move_bar():
    with pytest.raises(LawBlockedError):
        require_phase(intent="train_aim", move_contact_rate=0.5)


def test_phase_order_allows_aim_after_bar():
    require_phase(intent="train_aim", move_contact_rate=0.96)


def test_phase_order_refuses_selfplay_before_aim_delta():
    with pytest.raises(LawBlockedError):
        require_phase(
            intent="train_selfplay",
            move_contact_rate=0.99,
            aim_point_rate=0.9,
            aim_beats_move_vs_returner=False,
        )


def test_phase_order_allows_selfplay_after_aim_delta():
    require_phase(
        intent="train_selfplay",
        move_contact_rate=0.99,
        aim_point_rate=0.9,
        aim_beats_move_vs_returner=True,
    )


def test_claim_bans_refuse_lag_god():
    with pytest.raises(LawBlockedError):
        require_claims(lag_god=True)


def test_claim_bans_allow_clean():
    require_claims()


def test_scan_readme_current_is_clean():
    from pathlib import Path

    text = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(encoding="utf-8")
    flags = scan_text_flags(text)
    assert not any(flags.values()), flags

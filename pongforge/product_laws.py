# Copyright (c) 2026 Martial Systems LLC
"""Refuse laws. Verify-before-done is the finish gate."""

from __future__ import annotations

from typing import Any


def laws() -> list[dict[str, Any]]:
    from pongforge.graphs.claim_bans import build_graph as claim_bans
    from pongforge.graphs.phase_order import build_graph as phase_order

    return [
        {
            "id": "pong.phase_order",
            "build": phase_order,
            "state": {
                "intent": "train_move",
                "move_contact_rate": 1.0,
                "aim_point_rate": 1.0,
            },
            "allow_decisions": ["allow"],
        },
        {
            "id": "pong.claim_bans",
            "build": claim_bans,
            "state": {
                "lag_god": False,
                "centering_is_god": False,
                "selfplay_live_100_is_skill": False,
                "hemibrain_search": False,
                "neurokernel_play": False,
            },
            "allow_decisions": ["allow"],
        },
    ]

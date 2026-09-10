# Copyright (c) 2026 Martial Systems LLC
"""A before B before C."""

from __future__ import annotations

from typing import Any

from pongforge.graphs._common import binary_graph

CONTACT_BAR = 0.95
POINT_BAR = 0.80


def _evaluate(state: dict[str, Any]) -> dict[str, Any]:
    v: list[str] = []
    intent = str(state.get("intent") or "")
    contact = float(state.get("move_contact_rate") or 0.0)
    point = float(state.get("aim_point_rate") or 0.0)
    if intent in ("train_aim", "write_aim"):
        if contact < CONTACT_BAR:
            v.append("aim_before_move_bar")
    if intent in ("train_selfplay", "write_selfplay"):
        if contact < CONTACT_BAR:
            v.append("selfplay_before_move_bar")
        if point < POINT_BAR:
            v.append("selfplay_before_aim_bar")
    return {"violations": v, "events": [{"node": "evaluate", "ok": not v}]}


def build_graph():
    return binary_graph(
        name="pong.phase_order",
        evaluate=_evaluate,
        extra=["intent", "move_contact_rate", "aim_point_rate"],
    )

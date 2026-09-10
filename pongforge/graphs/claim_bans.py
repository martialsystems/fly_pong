# Copyright (c) 2026 Martial Systems LLC
"""Refuse god-claims on the lag bot and 100% vs a live copy."""

from __future__ import annotations

from typing import Any

from pongforge.graphs._common import binary_graph

_FLAGS = (
    "lag_god",
    "centering_is_god",
    "selfplay_live_100_is_skill",
    "hemibrain_search",
    "neurokernel_play",
)


def _evaluate(state: dict[str, Any]) -> dict[str, Any]:
    v = [k for k in _FLAGS if state.get(k)]
    return {"violations": v, "events": [{"node": "evaluate", "ok": not v}]}


def build_graph():
    return binary_graph(name="pong.claim_bans", evaluate=_evaluate, extra=list(_FLAGS))

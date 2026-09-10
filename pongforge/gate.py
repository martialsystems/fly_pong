# Copyright (c) 2026 Martial Systems LLC
"""Call sites for refuse laws."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pongforge._bootstrap import ensure_paths

ensure_paths()

from graphforge.product_law import LawBlockedError, require_law

from pongforge.graphs.claim_bans import build_graph as build_claims
from pongforge.graphs.phase_order import CONTACT_BAR, POINT_BAR, build_graph as build_phase

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"

GOD_RE = re.compile(
    r"\b(pong god|finished god|has mastered pong|100% vs (yourself|a live copy))\b",
    re.I,
)
LAG_GOD_RE = re.compile(r"\b(god)\b.*\b(lag|centering|99/100|40/40)\b", re.I)


def require_phase(
    *,
    intent: str,
    move_contact_rate: float,
    aim_point_rate: float = 0.0,
    aim_beats_move_vs_returner: bool = False,
) -> None:
    require_law(
        build_phase(),
        {
            "intent": intent,
            "move_contact_rate": float(move_contact_rate),
            "aim_point_rate": float(aim_point_rate),
            "aim_beats_move_vs_returner": bool(aim_beats_move_vs_returner),
        },
        allow_decisions=["allow"],
        law_id="pong.phase_order",
        thread_id=intent,
        raise_error=True,
    )


def require_claims(**flags: Any) -> None:
    thread_id = str(flags.pop("thread_id", "pong_claims"))
    state = {
        "lag_god": False,
        "centering_is_god": False,
        "selfplay_live_100_is_skill": False,
        "hemibrain_search": False,
        "neurokernel_play": False,
    }
    state.update(flags)
    require_law(
        build_claims(),
        state,
        allow_decisions=["allow"],
        law_id="pong.claim_bans",
        thread_id=thread_id,
        raise_error=True,
    )


def scan_text_flags(text: str) -> dict[str, bool]:
    t = text
    return {
        "lag_god": bool(LAG_GOD_RE.search(t) or re.search(r"lag.*god|god vs the lag", t, re.I)),
        "centering_is_god": bool(re.search(r"centering.*\b(god|mastered)\b", t, re.I)),
        "selfplay_live_100_is_skill": bool(
            re.search(r"100%.{0,40}(live copy|yourself|self-play copy).{0,20}skill", t, re.I)
        ),
        "hemibrain_search": bool(re.search(r"search.{0,20}hemibrain|140k neuron", t, re.I)),
        "neurokernel_play": bool(re.search(r"neurokernel.{0,20}(play|pong)", t, re.I)),
    }


def require_readme_clean(text: str) -> None:
    require_claims(**scan_text_flags(text), thread_id="readme")


def load_move_contact() -> float:
    path = LOGS / "device_move.json"
    if not path.is_file():
        return 0.0
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("contact_rate") or 0.0)


def load_aim_point() -> float:
    path = LOGS / "device_aim.json"
    if not path.is_file():
        return 0.0
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("point_rate") or 0.0)


def require_can_train_aim() -> None:
    require_phase(intent="train_aim", move_contact_rate=load_move_contact())


def load_aim_beats_returner() -> bool:
    path = LOGS / "aim_vs_returner.json"
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("passed")) and str(data.get("selfplay") or "") == "unlocked"


def require_can_train_selfplay() -> None:
    require_phase(
        intent="train_selfplay",
        move_contact_rate=load_move_contact(),
        aim_point_rate=load_aim_point(),
        aim_beats_move_vs_returner=load_aim_beats_returner(),
    )


__all__ = [
    "CONTACT_BAR",
    "POINT_BAR",
    "LawBlockedError",
    "require_phase",
    "require_claims",
    "require_readme_clean",
    "require_can_train_aim",
    "require_can_train_selfplay",
    "scan_text_flags",
]

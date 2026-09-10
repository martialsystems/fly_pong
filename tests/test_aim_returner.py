from __future__ import annotations

import json
from pathlib import Path

from fly_pong.features import AIM_KEYS

ROOT = Path(__file__).resolve().parents[1]


def test_hypothesis_was_written_before_results():
    hyp = json.loads((ROOT / "logs" / "aim_hypothesis.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "logs" / "aim_vs_returner.json").read_text(encoding="utf-8"))
    assert hyp["written_before_run"] is True
    assert hyp["pass"]["open_hit_delta_min"] == 0.08
    assert hyp["pass"]["point_delta_min"] == 40
    assert "predicted_contact_t" not in AIM_KEYS
    assert res["delta_agent_points"] == 52
    assert res["passed"] is True
    assert res["selfplay"] == "locked"
    assert res["move_aim"]["open_hit_rate"] < 0.65
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "316-321" in text
    assert "264-289" in text
    assert "Point delta +52 vs the freeze is not placement" in text
    assert "Self-play is not part of this title" in text
    assert hyp["log_line"].split(".")[0] in text
    assert hyp["log_line"] in agents

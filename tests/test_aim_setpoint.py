from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_setpoint_hypothesis_and_lock():
    hyp = json.loads((ROOT / "logs" / "aim_setpoint_hypothesis.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "logs" / "aim_setpoint.json").read_text(encoding="utf-8"))
    assert hyp["written_before_run"] is True
    assert hyp["control"]["window_frames"] == 24
    assert hyp["pass"]["scoring_key"] == "matches"
    assert hyp["pass"]["matches_min"] == 13
    assert hyp["pass"]["leak_max"] == 0.08
    assert res["passed"] is False
    assert res["leak_ok"] is False
    assert res["geo_ok"] is True
    assert res["score_ok"] is False
    assert res["write_c_hypothesis"] is False
    assert res["selfplay"] == "locked"
    assert res["move_aim"]["wins"] < 13
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert hyp["log_line"] in agents
    assert "7/40" in text
    assert "Leak is late arrival at the commanded Y (0.95)" in text
    from pongforge.gate import scan_text_flags

    assert not any(scan_text_flags(text).values())

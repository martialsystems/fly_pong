from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_supervised_sign_hypothesis_and_lock():
    hyp = json.loads((ROOT / "logs" / "aim_sign_supervised_hypothesis.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "logs" / "aim_sign_supervised.json").read_text(encoding="utf-8"))
    assert hyp["written_before_run"] is True
    assert hyp["train"]["widen_window"] is False
    assert hyp["pass"]["signed_open_geo_min"] == 0.569
    assert hyp["pass"]["leak_max"] == 0.08
    assert res["move_aim"]["signed_open"] >= 0.569
    assert res["leak_cmd_minus_geo"] > 0.08
    assert res["passed"] is False
    assert res["aim_retired_as_head"] is False
    assert res["selfplay"] == "locked"
    assert res["delta_open_hit"] >= 0.08
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert hyp["log_line"] in agents
    assert "0.814" in text or "0.814" in json.dumps(res["move_aim"]["signed_open"])
    assert "Self-play is not part of this title" in text
    assert "Self-play stays locked" in agents
    from pongforge.gate import scan_text_flags

    assert not any(scan_text_flags(text).values())

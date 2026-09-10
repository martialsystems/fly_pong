from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_landing_commit_hypothesis_and_lock():
    hyp = json.loads((ROOT / "logs" / "landing_commit_hypothesis.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "logs" / "landing_commit.json").read_text(encoding="utf-8"))
    assert hyp["written_before_run"] is True
    assert hyp["title"] == "commit-if-reachable"
    assert hyp["rule"]["on_commit"].startswith("bang-bang to intercept Y")
    assert "desired_offset" in hyp["rule"]["on_commit"]
    assert hyp["pass"]["scoring_key"] == "matches"
    assert hyp["pass"]["matches_min"] == 13
    assert hyp["pass"]["leak_max"] == 0.08
    assert hyp["pass"]["signed_open_geo_min"] == 0.80
    assert hyp["write_c_hypothesis"] is False
    assert res["passed"] is False
    assert res["leak_ok"] is True
    assert res["geo_ok"] is False
    assert res["score_ok"] is True
    assert res["write_c_hypothesis"] is False
    assert res["selfplay"] == "locked"
    assert res["commit"]["wins"] == 17
    assert res["commit"]["leak_rate"] == 0.0
    assert res["commit"]["signed_open"] < 0.80
    assert res["commit"]["abort_share"] > 0.0
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert hyp["log_line"] in agents
    assert "17/40" in text
    assert "0.465" in text
    assert "one veto, two names" in text.lower()
    assert "fruit landing = aim" not in text.lower()
    assert "## Locked numbers" in text
    assert "Play the PPO" not in text
    from pongforge.gate import scan_text_flags

    assert not any(scan_text_flags(text).values())

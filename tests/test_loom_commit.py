from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_loom_commit_hypothesis_and_lock():
    hyp = json.loads((ROOT / "logs" / "loom_commit_hypothesis.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "logs" / "loom_commit.json").read_text(encoding="utf-8"))
    assert hyp["written_before_run"] is True
    assert hyp["rule"]["window_frames"] == 24
    assert hyp["pass"]["matches_min"] == 13
    assert hyp["pass"]["matches_not_below_move_only"] == 12
    assert hyp["write_c_hypothesis"] is False
    assert res["passed"] is True
    assert res["write_c_hypothesis"] is False
    assert res["selfplay"] == "locked"
    assert res["loom_commit"]["wins"] == 17
    assert res["loom_commit"]["leak_rate"] == 0.0
    assert res["loom_commit"]["wins"] >= 13
    assert res["loom_commit"]["wins"] >= 12
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert hyp["log_line"] in agents
    assert "17/40" in text
    assert "logs/loom_commit.json" in text
    assert "goalie veto" in text
    from pongforge.gate import scan_text_flags

    assert not any(scan_text_flags(text).values())

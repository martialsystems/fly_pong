from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_leak_autopsy_hypothesis_and_lock():
    hyp = json.loads((ROOT / "logs" / "aim_leak_hypothesis.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "logs" / "aim_leak.json").read_text(encoding="utf-8"))
    assert hyp["written_before_run"] is True
    assert hyp["prior"]["leak"] == 0.186
    assert res["n_leak"] == 1000
    assert res["contacts"] == 5372
    assert res["counts"]["control_late"] == 950
    assert res["physics_is_the_cap"] is False
    assert res["control_is_the_cap"] is True
    assert res["write_c_hypothesis"] is False
    assert res["selfplay"] == "locked"
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert hyp["log_line"] in agents
    assert "950 control_late" in text
    from pongforge.gate import scan_text_flags

    assert not any(scan_text_flags(text).values())

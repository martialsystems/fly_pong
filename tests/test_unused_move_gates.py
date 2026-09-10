from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_unused_move_gates_hypothesis_and_lock():
    hyp = json.loads((ROOT / "logs" / "unused_move_gates_hypothesis.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "logs" / "unused_move_gates.json").read_text(encoding="utf-8"))
    assert hyp["written_before_run"] is True
    assert hyp["train"]["overwrite_device_pt"] is False
    assert hyp["pre_register"]["error_y_mass_min_to_call_lost"] == 0.8
    assert hyp["write_c_hypothesis"] is False
    assert res["unused_vision_lost_to_error_y"] is True
    assert res["vs_lag"]["error_y_mass"] >= 0.8
    assert res["lag_40_40_is_title"] is False
    assert res["vs_lag"]["wins"] == 40
    assert res["write_c_hypothesis"] is False
    assert res["selfplay"] == "locked"
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert hyp["log_line"] in agents
    assert "unused vision lost to error_y" in text.lower()
    assert "0.973" in text
    assert "logs/unused_move_gates.json" in text
    assert "not unused-vision skill" in text.lower()
    assert "plant unchanged" in text
    assert "titles unchanged" in text
    from pongforge.gate import scan_text_flags

    assert not any(scan_text_flags(text).values())

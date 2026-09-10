from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sign_hypothesis_before_results():
    hyp = json.loads((ROOT / "logs" / "aim_sign_hypothesis.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "logs" / "aim_sign.json").read_text(encoding="utf-8"))
    assert hyp["written_before_run"] is True
    assert hyp["pass"]["signed_open_delta_min"] == 0.10
    assert hyp["pass"]["open_hit_delta_min"] == 0.08
    assert "point" not in hyp["pass"]["rule"].split("AND")[0].lower() or "Point totals are reported and are not a pass key" in hyp["pass"]["rule"]
    assert res["passed"] is False
    assert res["random_corner_smasher"] is True
    assert res["selfplay"] == "locked"
    assert res["delta_signed_open"] < 0.10
    assert res["delta_open_hit"] < 0.08
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "random corner-smasher" in text
    assert hyp["log_line"] in text
    assert "Self-play remains locked" in text
    from pongforge.gate import scan_text_flags

    assert not any(scan_text_flags(text).values())

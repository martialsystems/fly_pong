from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_quotes_device_locks():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    move = json.loads((ROOT / "logs" / "device_move.json").read_text(encoding="utf-8"))
    aim = json.loads((ROOT / "logs" / "device_aim.json").read_text(encoding="utf-8"))
    assert move["n"] == 40
    assert move["wins"] == 40
    assert move["contact_rate"] >= 0.95
    assert move["g_move"]["error_y"] > 0.9
    assert aim["n"] == 40
    assert aim["wins"] == 40
    assert aim["point_rate"] >= 0.80
    assert "0.992" in text
    assert "422-48" in text
    assert "420-47" in text
    assert "error_y 0.984" in text
    assert "predicted_contact_t 0.900" in text
    assert "logs/device_move.json" in text
    assert "logs/device_aim.json" in text
    assert "Self-play" in text
    from pongforge.gate import scan_text_flags

    flags = scan_text_flags(text)
    assert not any(flags.values()), flags

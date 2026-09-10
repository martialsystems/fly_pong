from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_quotes_lock_files():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    fly = json.loads((ROOT / "logs" / "fly_gate.json").read_text(encoding="utf-8"))
    motion = json.loads((ROOT / "logs" / "fly_gate_motion_only.json").read_text(encoding="utf-8"))
    assert fly["wins"] == 40
    assert fly["n"] == 40
    assert fly["win_rate"] == 1.0
    assert fly["agent_points"] == 439
    assert fly["opp_points"] == 46
    assert fly["hit_frame_cap"] == 1
    assert motion["wins"] == 0
    assert motion["n"] == 20
    assert motion["agent_points"] == 0
    assert motion["opp_points"] == 220
    assert motion["controller"] == "t4t5_motion_only"
    assert fly["controller"] == "t4t5_plus_retinotopic_centering"
    assert "0/20" in text
    assert "0-220" in text
    assert "40/40" in text
    assert "439-46" in text
    assert "logs/fly_gate.json" in text
    assert "logs/fly_gate_motion_only.json" in text
    assert "20,000-frame cap at 10-4" in text
    assert "## Locked numbers" in text
    assert "—" not in text
    assert "What it is not" not in text
    assert "What this is not" not in text
    assert ".venv/bin/python" in text
    assert "python -m fly_pong.run_fly" in text
    assert "eval_fly.py" in text
    assert "Can a fly-style vertical motion detector play Pong?" in text

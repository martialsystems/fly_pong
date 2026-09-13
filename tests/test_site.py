from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"


def test_no_browser_court_page():
    assert not (PUBLIC / "index.html").is_file()
    assert not (PUBLIC / "models" / "pong.onnx").is_file()
    assert not (PUBLIC / "js" / "game.js").is_file()


def test_readme_quality():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "—" not in text
    assert "What it is not" not in text
    assert "What this is not" not in text
    assert ".venv/bin/python" in text
    assert "python -m fly_pong.run_human" in text
    assert "--opponent approach" in text
    assert "Play the PPO" not in text
    assert "Do not start another aim head" not in text
    assert "Do not retcon" not in text
    assert "github.io/fly_pong" not in text
    assert "6-input PPO" not in text
    assert "not a fly result" in text
    assert "not a placement result" in text
    assert "Self-play is not part of this title" in text
    assert "[Fly research index](https://gist.github.com/martialsystems/12835f747d6360781f3cc7f91f243178)" in text
    assert "–" not in text
    assert "Do not start another aim head on this object" in agents
    assert "Do not retcon the +52 point OR-bar into aim" in agents
    assert "run_human --opponent approach" in agents
    hook = (ROOT / "description.txt").read_text(encoding="utf-8").strip()
    assert hook == (
        "T4/T5 + centering returns vs lag. Placement is closed. "
        "Play vs approach_commit in Python."
    )

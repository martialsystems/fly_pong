from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"


def test_public_has_no_fly_brain():
    banned = re.compile(r"T4|T5|FlyBrainLab|ommatid|Neurokernel", re.I)
    hits = []
    for path in PUBLIC.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".onnx", ".png", ".jpg", ".wasm"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if banned.search(text):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], f"fly-brain tokens in public/: {hits}"


def test_public_assets_exist():
    html = (PUBLIC / "index.html").read_text(encoding="utf-8")
    assert 'src="js/game.js"' in html
    assert 'href="css/style.css"' in html
    for rel in ("js/game.js", "js/physics.js", "js/obs.js", "js/ai.js", "js/constants.js", "css/style.css"):
        assert (PUBLIC / rel).is_file(), rel
    # relative assets named by HTML
    for m in re.finditer(r"""(?:src|href)=["']([^"']+)["']""", html):
        url = m.group(1)
        if url.startswith("http"):
            continue
        assert (PUBLIC / url).is_file(), url


def test_constants_js_matches_shared():
    shared = json.loads((ROOT / "shared" / "constants.json").read_text(encoding="utf-8"))
    public_json = json.loads((PUBLIC / "constants.json").read_text(encoding="utf-8"))
    assert shared == public_json
    js = (PUBLIC / "js" / "constants.js").read_text(encoding="utf-8")
    assert "generated from shared/constants.json" in js
    assert "do not edit" in js
    # pull the object literal
    blob = js.split("export const C = ", 1)[1].strip().rstrip(";")
    assert json.loads(blob) == shared


def test_readme_quality():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "—" not in text
    assert "What it is not" not in text
    assert "What this is not" not in text
    assert ".venv/bin/python" in text
    assert "python -m fly_pong.run_fly" in text or "python -m fly_pong.run_human" in text
    assert "Play the PPO" not in text
    assert "Do not start another aim head" not in text
    assert "Do not retcon" not in text
    assert "6-input PPO" in text
    assert "not a fly result" in text
    assert "not a placement result" in text
    assert "Self-play is not part of this title" in text
    assert "[Fly research index](https://gist.github.com/martialsystems/12835f747d6360781f3cc7f91f243178)" in text
    assert "–" not in text
    assert "Do not start another aim head on this object" in agents
    assert "Do not retcon the +52 point OR-bar into aim" in agents
    hook = (ROOT / "description.txt").read_text(encoding="utf-8").strip()
    assert hook == (
        "T4/T5 + centering won 40/40 vs lag. Placement is closed. "
        "The court page is a separate PPO."
    )


def test_page_labels_ppo_not_fly():
    html = (PUBLIC / "index.html").read_text(encoding="utf-8")
    js = (PUBLIC / "js" / "game.js").read_text(encoding="utf-8")
    assert "6-input PPO" in html
    assert "not the fly controller" in html
    assert "AI:" not in html
    assert "AI:" not in js
    assert "PPO (ONNX)" in js
    assert "lag-chase" in js

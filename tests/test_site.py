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
    assert "—" not in text
    assert "What it is not" not in text
    assert "What this is not" not in text
    assert ".venv/bin/python" in text
    assert "python -m fly_pong.run_fly" in text or "python -m fly_pong.run_human" in text

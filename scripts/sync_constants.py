#!/usr/bin/env python3
"""Copy shared/constants.json into the public JS module. Do not hand-edit the JS."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "shared" / "constants.json"
DST = ROOT / "public" / "js" / "constants.js"
JSON_DST = ROOT / "public" / "constants.json"


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    JSON_DST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    body = (
        "/* generated from shared/constants.json; do not edit */\n"
        "export const C = "
        + json.dumps(data, indent=2)
        + ";\n"
    )
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(body, encoding="utf-8")
    print(f"wrote {DST.relative_to(ROOT)} and {JSON_DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

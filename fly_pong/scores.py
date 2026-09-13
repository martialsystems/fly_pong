"""Local high scores for the Python court. Not a fly title."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / ".fly_pong" / "high_scores.json"
KEEP = 10


def load(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or DEFAULT_PATH
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "human": int(row.get("human") or 0),
                "fly": int(row.get("fly") or 0),
                "win": bool(row.get("win")),
            }
        )
    return out


def save(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    p = path or DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rows": rows[:KEEP]}, indent=2) + "\n", encoding="utf-8")


def record(human: int, fly: int, *, path: Path | None = None) -> list[dict[str, Any]]:
    rows = load(path)
    rows.append({"human": int(human), "fly": int(fly), "win": int(human) > int(fly)})
    rows.sort(key=lambda r: (int(r["win"]), int(r["human"]), -int(r["fly"])), reverse=True)
    rows = rows[:KEEP]
    save(rows, path)
    return rows

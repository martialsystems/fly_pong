"""Load the shared physics/observation contract."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_PATH = REPO_ROOT / "shared" / "constants.json"


@lru_cache(maxsize=1)
def load_constants() -> dict[str, Any]:
    with CONSTANTS_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    required = (
        "width",
        "height",
        "paddleW",
        "paddleH",
        "paddleSpeed",
        "ballR",
        "ballSpeed",
        "maxScore",
        "velScale",
        "agentX",
        "oppX",
        "bounceGain",
        "spin",
        "oppLag",
        "serveAngleMax",
        "fps",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"shared/constants.json missing keys: {missing}")
    return data

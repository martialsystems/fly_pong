#!/usr/bin/env python3
"""Phase C: self-play vs a frozen clone. Requires aim point bar."""

from __future__ import annotations

import argparse
from pathlib import Path

from fly_pong.device import FlyPongDevice
from pongforge.gate import require_can_train_selfplay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ARTIFACTS / "device.pt")
    args = parser.parse_args()
    require_can_train_selfplay()
    print(
        "self-play trainer is gated and ready. "
        "Run eval_device.py --opponent clone after copying weights to a frozen clone."
    )
    if args.weights.is_file():
        FlyPongDevice().load(args.weights)
        print(f"loaded {args.weights}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Refuse paths for pongforge laws."""

from __future__ import annotations

from pongforge.gate import LawBlockedError, require_claims, require_phase


def main() -> None:
    require_phase(intent="train_move", move_contact_rate=0.0)
    try:
        require_phase(intent="train_aim", move_contact_rate=0.1)
        raise SystemExit("expected aim-before-move block")
    except LawBlockedError:
        pass
    try:
        require_claims(lag_god=True)
        raise SystemExit("expected lag_god block")
    except LawBlockedError:
        pass
    require_claims()
    print("pongforge sanity pass")


if __name__ == "__main__":
    main()
